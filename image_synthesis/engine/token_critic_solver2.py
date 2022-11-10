# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import time
import math
import torch
import torch.nn.functional as F
import threading
import multiprocessing
import copy
from PIL import Image
from torch.nn.utils import clip_grad_norm_, clip_grad_norm
import torchvision
from image_synthesis.utils.misc import instantiate_from_config, format_seconds
from image_synthesis.distributed.distributed import reduce_dict
from image_synthesis.distributed.distributed import is_primary, get_rank
from image_synthesis.utils.misc import get_model_parameters_info
from image_synthesis.engine.lr_scheduler import ReduceLROnPlateauWithWarmup, CosineAnnealingLRWithWarmup
from image_synthesis.engine.ema import EMA
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics.image.fid import FrechetInceptionDistance
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP = True
except:
    print('Warning: import torch.amp failed, so no amp will be used!')
    AMP = False

import scipy.linalg
import numpy as np
from nsml import IS_ON_NSML
from nsml_utils import bind_model
import nsml

STEP_WITH_LOSS_SCHEDULERS = (ReduceLROnPlateauWithWarmup, ReduceLROnPlateau)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

class Token_Critic_Solver(object):
    def __init__(self, config, args, token_critic_model, diffusion_model, dataloader, logger):
        self.config = config
        self.args = args
        self.model = token_critic_model 
        self.dataloader = dataloader
        self.logger = logger
        self.vq_diffusion = diffusion_model
        self.use_my_ckpt = args.use_my_ckpt
        # if self.model.learnable_cf:
        #     self.empty_text_embed = self.model.empty_text_embed

        self.max_epochs = config['solver']['max_epochs']
        self.save_epochs = config['solver']['save_epochs']
        self.save_iterations = config['solver'].get('save_iterations', -1)
        self.sample_iterations = config['solver']['sample_iterations']
        if self.sample_iterations == 'epoch':
            self.sample_iterations = self.dataloader['train_iterations']
        self.validation_epochs = config['solver'].get('validation_epochs', 2)
        assert isinstance(self.save_epochs, (int, list))
        assert isinstance(self.validation_epochs, (int, list))
        self.debug = config['solver'].get('debug', False)

        self.last_epoch = -1
        self.last_iter = -1
        self.ckpt_dir = os.path.join(args.save_dir, 'checkpoint')
        self.image_dir = os.path.join(args.save_dir, 'images')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        # get grad_clipper
        if 'clip_grad_norm' in config['solver']:
            self.clip_grad_norm = instantiate_from_config(config['solver']['clip_grad_norm'])
        else:
            self.clip_grad_norm = None

        # get lr
        adjust_lr = config['solver'].get('adjust_lr', 'sqrt')
        base_lr = config['solver'].get('base_lr', 1.0e-4)
        if adjust_lr == 'none':
            self.lr = base_lr
        elif adjust_lr == 'sqrt':
            self.lr = base_lr * math.sqrt(args.world_size * config['dataloader']['batch_size'])
        elif adjust_lr == 'linear':
            self.lr = base_lr * args.world_size * config['dataloader']['batch_size']
        else:
            raise NotImplementedError('Unknown type of adjust lr {}!'.format(adjust_lr))
        self.logger.log_info('Get lr {} from base lr {} with {}'.format(self.lr, base_lr, adjust_lr))

        if hasattr(token_critic_model, 'get_optimizer_and_scheduler') and callable(getattr(token_critic_model, 'get_optimizer_and_scheduler')):
            optimizer_and_scheduler = token_critic_model.get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])
        else:
            optimizer_and_scheduler = self._get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])

        assert type(optimizer_and_scheduler) == type({}), 'optimizer and schduler should be a dict!'
        self.optimizer_and_scheduler = optimizer_and_scheduler

        # configre for ema
        if 'ema' in config['solver'] and args.local_rank == 0:
            ema_args = config['solver']['ema']
            ema_args['model'] = self.model
            self.ema = EMA(**ema_args)
        else:
            self.ema = None

        self.num_classes = self.vq_diffusion.transformer.num_classes
        self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.model.cuda()
        self.vq_diffusion.cuda()
        
        self.device = self.model.device
        if self.args.distributed:
            self.logger.log_info('Distributed, begin DDP the model...')
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu], find_unused_parameters=False)
            self.vq_diffusion = torch.nn.parallel.DistributedDataParallel(self.vq_diffusion, device_ids=[self.args.gpu], find_unused_parameters=False)
            self.logger.log_info('Distributed, DDP model done!')
        # prepare for amp
        self.args.amp = self.args.amp and AMP
        if self.args.amp:
            self.scaler = GradScaler()
            self.logger.log_info('Using AMP for training!')

        if self.args.only_val:
            self.fid = FrechetInceptionDistance(2048).to(self.device)

        self.logger.log_info("{}: global rank {}: prepare solver done!".format(self.args.name,self.args.global_rank), check_primary=False)

        
        
        self.n_sample = [64] * 16
        self.time_list = [index for index in range(100 -5, -1, -6)]

    def _get_optimizer_and_scheduler(self, op_sc_list):
        optimizer_and_scheduler = {}
        for op_sc_cfg in op_sc_list:
            op_sc = {
                'name': op_sc_cfg.get('name', 'none'),
                'start_epoch': op_sc_cfg.get('start_epoch', 0),
                'end_epoch': op_sc_cfg.get('end_epoch', -1),
                'start_iteration': op_sc_cfg.get('start_iteration', 0),
                'end_iteration': op_sc_cfg.get('end_iteration', -1),
            }

            if op_sc['name'] == 'none':
                # parameters = self.model.parameters()
                parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            else:
                # NOTE: get the parameters with the given name, the parameters() should be overide
                parameters = self.model.parameters(name=op_sc['name'])
            
            # build optimizer
            op_cfg = op_sc_cfg.get('optimizer', {'target': 'torch.optim.SGD', 'params': {}})
            if 'params' not in op_cfg:
                op_cfg['params'] = {}
            if 'lr' not in op_cfg['params']:
                op_cfg['params']['lr'] = self.lr
            op_cfg['params']['params'] = parameters
            optimizer = instantiate_from_config(op_cfg)
            op_sc['optimizer'] = {
                'module': optimizer,
                'step_iteration': op_cfg.get('step_iteration', 1)
            }
            assert isinstance(op_sc['optimizer']['step_iteration'], int), 'optimizer steps should be a integer number of iterations'

            # build scheduler
            if 'scheduler' in op_sc_cfg:
                sc_cfg = op_sc_cfg['scheduler']
                sc_cfg['params']['optimizer'] = optimizer
                # for cosine annealing lr, compute T_max
                if sc_cfg['target'].split('.')[-1] in ['CosineAnnealingLRWithWarmup', 'CosineAnnealingLR']:
                    T_max = self.max_epochs * self.dataloader['train_iterations']
                    sc_cfg['params']['T_max'] = T_max
                scheduler = instantiate_from_config(sc_cfg)
                op_sc['scheduler'] = {
                    'module': scheduler,
                    'step_iteration': sc_cfg.get('step_iteration', 1)
                }
                if op_sc['scheduler']['step_iteration'] == 'epoch':
                    op_sc['scheduler']['step_iteration'] = self.dataloader['train_iterations']
            optimizer_and_scheduler[op_sc['name']] = op_sc

        return optimizer_and_scheduler

    def _get_lr(self, return_type='str'):
        
        lrs = {}
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            lr = op_sc['optimizer']['module'].state_dict()['param_groups'][0]['lr']
            lrs[op_sc_n+'_lr'] = round(lr, 10)
        if return_type == 'str':
            lrs = str(lrs)
            lrs = lrs.replace('none', 'lr').replace('{', '').replace('}','').replace('\'', '')
        elif return_type == 'dict':
            pass 
        else:
            raise ValueError('Unknow of return type: {}'.format(return_type))
        return lrs

    def sample(self, batch, phase='train', step_type='iteration'):
        tic = time.time()
    
    def real_mask_recon(self, data_i, truncation_rate, guidance_scale=5.0):
        if self.args.distributed:
            vq_diffusion = self.vq_diffusion.module
        else:
            vq_diffusion = self.vq_diffusion

        with torch.no_grad():
            model_out = vq_diffusion.real_mask_return(
                batch=data_i,
                filter_ratio=0,
                content_ratio=1,
                return_att_weight=False,
                truncation_rate=truncation_rate,
            ) # {'t', 'changed', 'recon_token'}

        return model_out

    def step(self, batch, phase='train'):

        if self.args.distributed:
            vq_diffusion = self.vq_diffusion
        else:
            vq_diffusion = self.vq_diffusion.module

        loss = {}
        if self.debug == False: 
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()
        else:
            batch = batch[0].cuda()

        with torch.no_grad():
                vq_out = self.real_mask_recon(
                    data_i = batch,
                    truncation_rate = 0.86,
                    guidance_scale = 5.0
                )

        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            if phase == 'train':
                # check if this optimizer and scheduler is valid in this iteration and epoch
                if op_sc['start_iteration'] > self.last_iter:
                    continue
                if op_sc['end_iteration'] > 0 and op_sc['end_iteration'] <= self.last_iter:
                    continue
                if op_sc['start_epoch'] > self.last_epoch:
                    continue
                if op_sc['end_epoch'] > 0 and op_sc['end_epoch'] <= self.last_epoch:
                    continue

            input = {
                'batch': batch,
                'return_loss': True,
                'step': self.last_iter,
                }
            if op_sc_n != 'none':
                input['name'] = op_sc_n

            if phase == 'train':
                if self.args.amp:
                    with autocast():
                        _, output = self.model(batch['text'], vq_out)
                else:
                    _, output = self.model(batch['text'], vq_out)
            else:
                with torch.no_grad():
                    if self.args.amp:
                        with autocast():
                            _, output = self.model(batch['text'], vq_out)
                    else:
                        _, output = self.model(batch['text'], vq_out)
            output = {'loss': output, }
            if phase == 'train':
                if op_sc['optimizer']['step_iteration'] > 0 and (self.last_iter + 1) % op_sc['optimizer']['step_iteration'] == 0:
                    op_sc['optimizer']['module'].zero_grad()
                    if self.args.amp:
                        self.scaler.scale(output['loss']).backward()
                        if self.clip_grad_norm is not None:
                            self.clip_grad_norm(self.model.parameters())
                        self.scaler.step(op_sc['optimizer']['module'])
                        self.scaler.update()
                    else:
                        output['loss'].backward()
                        if self.clip_grad_norm is not None:
                            self.clip_grad_norm(self.model.parameters())
                        op_sc['optimizer']['module'].step()
                    
                if 'scheduler' in op_sc:
                    if op_sc['scheduler']['step_iteration'] > 0 and (self.last_iter + 1) % op_sc['scheduler']['step_iteration'] == 0:
                        if isinstance(op_sc['scheduler']['module'], STEP_WITH_LOSS_SCHEDULERS):
                            op_sc['scheduler']['module'].step(output.get('loss'))
                        else:
                            op_sc['scheduler']['module'].step()
                # update ema model
                if self.ema is not None:
                    self.ema.update(iteration=self.last_iter)

            loss[op_sc_n] = {k: v for k, v in output.items() if ('loss' in k or 'acc' in k)}
        return loss

    def save(self, force=False):
        if is_primary():
            # save with the epoch specified name
            if self.save_iterations > 0:
                if (self.last_iter + 1) % self.save_iterations == 0:
                    save = True 
                else:
                    save = False
            else:
                if isinstance(self.save_epochs, int):
                    save = (self.last_epoch + 1) % self.save_epochs == 0
                else:
                    save = (self.last_epoch + 1) in self.save_epochs
                
            if save or force:
                if IS_ON_NSML:
                    bind_model(self.last_epoch, self.last_iter, self.model, self.ema, self.clip_grad_norm, self.optimizer_and_scheduler, self.args.local_rank, None, None)
                    nsml.save(self.last_epoch)
                    print("saved epoch ", self.last_epoch)
                else:
                    state_dict = {
                        'last_epoch': self.last_epoch,
                        'last_iter': self.last_iter,
                        'model': self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict() 
                    }
                    if self.ema is not None:
                        state_dict['ema'] = self.ema.state_dict()
                    if self.clip_grad_norm is not None:
                        state_dict['clip_grad_norm'] = self.clip_grad_norm.state_dict()

                    # add optimizers and schedulers
                    optimizer_and_scheduler = {}
                    for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
                        state_ = {}
                        for k in op_sc:
                            if k in ['optimizer', 'scheduler']:
                                op_or_sc = {kk: vv for kk, vv in op_sc[k].items() if kk != 'module'}
                                op_or_sc['module'] = op_sc[k]['module'].state_dict()
                                state_[k] = op_or_sc
                            else:
                                state_[k] = op_sc[k]
                        optimizer_and_scheduler[op_sc_n] = state_

                    state_dict['optimizer_and_scheduler'] = optimizer_and_scheduler
                    if save:
                        save_path = os.path.join(self.ckpt_dir, '{}e_{}iter.pth'.format(str(self.last_epoch).zfill(6), self.last_iter))
                        torch.save(state_dict, save_path)
            
                    self.logger.log_info('saved in {}'.format(save_path))    
                
                # save with the last name
                if not IS_ON_NSML:
                    save_path = os.path.join(self.ckpt_dir, 'last.pth')
                    torch.save(state_dict, save_path)  
                    self.logger.log_info('saved in {}'.format(save_path))     
        
    def resume(self, 
               path=None, # The path of last.pth
               load_optimizer_and_scheduler=True, # whether to load optimizers and scheduler
               load_others=True # load other informations
               ): 
        if path is None:
            path = os.path.join(self.ckpt_dir, 'last.pth')

        if IS_ON_NSML and self.use_my_ckpt:
            path = "ailab002/kaist_coco_vq/" + path
            bind_model(self.last_epoch, self.last_iter, self.model, self.ema, self.clip_grad_norm, self.optimizer_and_scheduler, self.args.local_rank, load_others, load_optimizer_and_scheduler)
            resume_info = path.rsplit('/', 1)
            session_name = resume_info[0]
            checkpoint_num = resume_info[1]
            nsml.load(checkpoint=checkpoint_num, session=session_name)
            print("loaded ", resume_info)
        else:
            state_dict = torch.load(path, map_location='cuda:{}'.format(self.args.local_rank))

            if load_others:
                self.last_epoch = state_dict['last_epoch']
                self.last_iter = state_dict['last_iter']
            
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                try:
                    self.model.module.load_state_dict(state_dict['model'])
                except:
                    model_dict = self.model.module.state_dict()
                    temp_state_dict = {k:v for k,v in state_dict['model'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(state_dict['model'])

            if 'ema' in state_dict and self.ema is not None:
                try:
                    self.ema.load_state_dict(state_dict['ema'])
                except:
                    model_dict = self.ema.state_dict()
                    temp_state_dict = {k:v for k,v in state_dict['ema'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.ema.load_state_dict(model_dict)

            if 'clip_grad_norm' in state_dict and self.clip_grad_norm is not None and state_dict['clip_grad_norm'] is not None:
                self.clip_grad_norm.load_state_dict(state_dict['clip_grad_norm'])

            # handle optimizer and scheduler
            if state_dict['optimizer_and_scheduler'] is not None:
                for op_sc_n, op_sc in state_dict['optimizer_and_scheduler'].items():
                    for k in op_sc:
                        if k in ['optimizer', 'scheduler']:
                            for kk in op_sc[k]:
                                if kk == 'module' and load_optimizer_and_scheduler:
                                    self.optimizer_and_scheduler[op_sc_n][k][kk].load_state_dict(op_sc[k][kk])
                                elif load_others: # such as step_iteration, ...
                                    self.optimizer_and_scheduler[op_sc_n][k][kk] = op_sc[k][kk]
                        elif load_others: # such as start_epoch, end_epoch, ....
                            self.optimizer_and_scheduler[op_sc_n][k] = op_sc[k]


        """
        VQ load
        """
        state_dict = torch.load(self.args.vq_path, map_location='cuda:{}'.format(self.args.local_rank))
        missing, unexpected = self.vq_diffusion.module.load_state_dict(state_dict['model'])
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)
    
        
        self.logger.log_info('Resume from {}'.format(path))

    
    def train_epoch(self):
        self.model.train()
        self.last_epoch += 1

        if self.args.distributed:
            self.dataloader['train_loader'].sampler.set_epoch(self.last_epoch)

        epoch_start = time.time()
        itr_start = time.time()
        itr = -1
        for itr, batch in enumerate(self.dataloader['train_loader']):
            if itr == 0:
                print("time2 is " + str(time.time()))
            data_time = time.time() - itr_start
            step_start = time.time()
            self.last_iter += 1
            loss = self.step(batch, phase='train')
            # logging info
            if self.logger is not None and self.last_iter % self.args.log_frequency == 0:
                info = '{}: train'.format(self.args.name)
                info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                for loss_n, loss_dict in loss.items():
                    info += ' ||'
                    loss_dict = reduce_dict(loss_dict)
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    # info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(tag='train/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]), global_step=self.last_iter)
                
                # log lr
                lrs = self._get_lr(return_type='dict')
                for k in lrs.keys():
                    lr = lrs[k]
                    self.logger.add_scalar(tag='train/{}_lr'.format(k), scalar_value=lrs[k], global_step=self.last_iter)

                # add lr to info
                info += ' || {}'.format(self._get_lr())
                    
                # add time consumption to info
                spend_time = time.time() - self.start_train_time
                itr_time_avg = spend_time / (self.last_iter + 1)
                info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | iter_avg_time: {ita}s | epoch_time: {et} | spend_time: {st} | left_time: {lt}'.format(
                        dt=round(data_time, 1),
                        it=round(time.time() - itr_start, 1),
                        fbt=round(time.time() - step_start, 1),
                        ita=round(itr_time_avg, 1),
                        et=format_seconds(time.time() - epoch_start),
                        st=format_seconds(spend_time),
                        lt=format_seconds(itr_time_avg*self.max_epochs*self.dataloader['train_iterations']-spend_time)
                        )
                self.logger.log_info(info)
            
            itr_start = time.time()

            # sample
            # if self.sample_iterations > 0 and (self.last_iter + 1) % self.sample_iterations == 0:
            #     # print("save model here")
            #     # self.save(force=True)
            #     # print("save model done")
            #     self.model.eval()
            #     self.sample(batch, phase='train', step_type='iteration')
            #     self.model.train()

        # modify here to make sure dataloader['train_iterations'] is correct
        assert itr >= 0, "The data is too less to form one iteration!"
        self.dataloader['train_iterations'] = itr + 1

    def validate_epoch(self):
        if 'validation_loader' not in self.dataloader:
            val = False
        else:
            if isinstance(self.validation_epochs, int):
                val = (self.last_epoch + 1) % self.validation_epochs == 0
            else:
                val = (self.last_epoch + 1) in self.validation_epochs        
        
        if val:
            if self.args.distributed:
                self.dataloader['validation_loader'].sampler.set_epoch(self.last_epoch)

            self.model.eval()
            overall_loss = None
            epoch_start = time.time()
            itr_start = time.time()
            itr = -1
            for itr, batch in enumerate(self.dataloader['validation_loader']):
                data_time = time.time() - itr_start
                step_start = time.time()
                loss = self.step(batch, phase='val')
                
                for loss_n, loss_dict in loss.items():
                    loss[loss_n] = reduce_dict(loss_dict)
                if overall_loss is None:
                    overall_loss = loss
                else:
                    for loss_n, loss_dict in loss.items():
                        for k, v in loss_dict.items():
                            overall_loss[loss_n][k] = (overall_loss[loss_n][k] * itr + loss[loss_n][k]) / (itr + 1)
                
                if self.logger is not None and (itr+1) % self.args.log_frequency == 0:
                    info = '{}: val'.format(self.args.name) 
                    info = info + ': Epoch {}/{} | iter {}/{}'.format(self.last_epoch, self.max_epochs, itr, self.dataloader['validation_iterations'])
                    for loss_n, loss_dict in loss.items():
                        info += ' ||'
                        info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                        # info = info + ': Epoch {}/{} | iter {}/{}'.format(self.last_epoch, self.max_epochs, itr, self.dataloader['validation_iterations'])
                        for k in loss_dict:
                            info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        
                    itr_time_avg = (time.time() - epoch_start) / (itr + 1)
                    info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | epoch_time: {et} | left_time: {lt}'.format(
                            dt=round(data_time, 1),
                            fbt=round(time.time() - step_start, 1),
                            it=round(time.time() - itr_start, 1),
                            et=format_seconds(time.time() - epoch_start),
                            lt=format_seconds(itr_time_avg*(self.dataloader['train_iterations']-itr-1))
                            )
                        
                    self.logger.log_info(info)
                itr_start = time.time()
            # modify here to make sure dataloader['validation_iterations'] is correct
            assert itr >= 0, "The data is too less to form one iteration!"
            self.dataloader['validation_iterations'] = itr + 1

            if self.logger is not None:
                info = '{}: val'.format(self.args.name) 
                for loss_n, loss_dict in overall_loss.items():
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    info += ': Epoch {}/{}'.format(self.last_epoch, self.max_epochs)
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(tag='val/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]), global_step=self.last_epoch)
                self.logger.log_info(info)

    def n_t(self, t, a=0.1, b=0.2):
        # t smaller(initial stage) -> variance bigger
        p = t / 100 * a + b
        return p

    @torch.no_grad()
    def generate_token_critic_for_metric(self, batch):
        if self.args.distributed:
            vq_diffusion = self.vq_diffusion.module
        else:
            vq_diffusion = self.vq_diffusion
            
        batch_size = len(batch['text'])

        with torch.no_grad(): # condition(CLIP) -> freeze
            condition_token = vq_diffusion.condition_codec.get_tokens(batch['text']) # BPE token
            cond_ = {}
            for k, v in condition_token.items():
                v = v.to(self.device) if torch.is_tensor(v) else v
                cond_['condition_' + k] = v
            cond_emb = vq_diffusion.transformer.condition_emb(cond_['condition_token']).float() # CLIP condition
            # tc_cf_cond_emb = self.empty_text_embed

        zero_logits = torch.zeros((batch_size, self.num_classes-1, vq_diffusion.transformer.shape),device=self.device)
        one_logits = torch.ones((batch_size, 1, vq_diffusion.transformer.shape),device=self.device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        with torch.no_grad():
            for i, (n, diffusion_index) in enumerate(zip(self.n_sample[:-1], self.time_list[:-1])): # before last step
                # 1) VQ: 1 step reconstruction
                t = torch.full((batch_size,), diffusion_index, device=self.device, dtype=torch.long)
                _, log_x_recon = vq_diffusion.transformer.p_pred(log_z, cond_emb, t)
                out = vq_diffusion.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024
                out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list

                # 2) TC: Masking based on score
                t_1 = torch.full((batch_size,), self.time_list[i+1], device=self.device, dtype=torch.long) # t-1 step

                if self.args.amp:
                    with autocast():
                        if self.args.distributed:
                            score = self.model.module.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                            if self.args.tc_guidance != None:
                                cf_score = self.model.module.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=self.empty_text_embed)
                                score = cf_score + self.args.tc_guidance * (score - cf_score)
                        else:
                            score = self.model.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                            if self.args.tc_guidance != None:
                                cf_score = self.model.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=self.empty_text_embed)
                                score = cf_score + self.args.tc_guidance * (score - cf_score)
                else:
                    if self.args.distributed:
                        score = self.model.module.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                        if self.args.tc_guidance != None:
                            cf_score = self.model.module.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=self.empty_text_embed)
                            score = cf_score + self.args.tc_guidance * (score - cf_score)
                    else:
                        score = self.model.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                        if self.args.tc_guidance != None:
                            cf_score = self.model.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=self.empty_text_embed)
                            score = cf_score + self.args.tc_guidance * (score - cf_score)
                                
                score = 1 - score # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                score = score.clamp(min = 0) # score clamp for CF minus probability
                score = (score - score.mean(1, keepdim=True)) * self.args.tc_weight + score.mean(1, keepdim=True)
                score = score.clamp(min = 0)
                if self.args.tc_det:
                    score += self.args.tc_alpha * diffusion_index / 100 * torch.rand_like(score).to(out_idx.device)
                else:
                    score += self.n_t(diffusion_index, self.args.tc_a, self.args.tc_b) # n(t) for randomness

                for ii in range(batch_size):
                    if self.args.tc_det:
                        _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                    else:
                        sel = torch.multinomial(score[ii], n)
                    out2_idx[ii][sel] = out_idx[ii][sel]
                log_z = index_to_log_onehot(out2_idx, self.num_classes)

            # Final step
            t = torch.full((batch_size,), self.time_list[-1], device=self.device, dtype=torch.long)
            _, log_x_recon = vq_diffusion.transformer.p_pred(log_z, cond_emb, t)
            out = vq_diffusion.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
            content_token = log_onehot_to_index(out) # b, 1024

            content = vq_diffusion.content_codec.decode(content_token) 
        return content

    def validate_epoch_fid(self):
        if 'validation_loader' not in self.dataloader:
            val = False
        else:
            if isinstance(self.validation_epochs, int):
                val = (self.last_epoch + 1) % self.validation_epochs == 0
            else:
                val = (self.last_epoch + 1) in self.validation_epochs        
        
        if val:
            if self.args.distributed:
                self.dataloader['validation_loader'].sampler.set_epoch(self.last_epoch)

            self.model.eval()
            self.vq_diffusion.eval()
            # num_batch = len(self.dataloader['validation_loader'])   # -> full
            num_batch = 625                                         # 625 -> 40k, 1250 -> 80k

            tot_batch = []
            tot_out = []
            for itr, batch in enumerate(self.dataloader['validation_loader']):
                if itr==num_batch:
                    del batch
                    break
                batch["image"] = batch["image"].to(self.device)
                with torch.no_grad():
                    output = self.generate_token_critic_for_metric(batch = batch)

                self.fid.update(batch["image"].type(torch.uint8), real=True)
                self.fid.update(output.type(torch.uint8), real=False)
                if self.args.local_rank==0:
                    print("[ ", itr, " / ", num_batch, " ]")
                del batch, output
            if self.args.local_rank==0:
                print("computing...")
            fid_score = self.fid.compute()
            if self.args.local_rank ==0:
                print("FID : ", float(fid_score))
                print("FID_real samples: ", self.fid.real_features_num_samples)
                print("FID_fake samples: ", self.fid.fake_features_num_samples)
                

    def validate(self):
        # setting for token_critic inference
        self.vq_diffusion.module.transformer.eval()
        self.model.module.transformer.eval()
        if self.args.distributed:
            vq_diffusion = self.vq_diffusion.module
        else:
            vq_diffusion = self.vq_diffusion
        # print(f"empty_text_embed: {self.model.module.empty_text_embed}")
        if self.args.tc_step == 16:
            self.n_sample = [64] * 16
            self.time_list = [index for index in range(100 -5, -1, -6)]
        elif self.args.tc_step == 50:
            self.n_sample = [10] + [21, 20] * 24 + [30]
            self.time_list = [index for index in range(100 -1, -1, -2)]
        else: # 100
            self.n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
            self.time_list = [index for index in range(100 -1, -1, -1)]

        for s in range(1, self.args.tc_step):
            self.n_sample[s] += self.n_sample[s-1]
        # setting for CF_guidance vq diffusion

        batch_size = self.args.batch_size
        cf_cond_emb = self.vq_diffusion.module.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.vq_diffusion.module.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(self.vq_diffusion.module.guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.vq_diffusion.module.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.vq_diffusion.module.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
            log_new_x_recon = cf_log_x_recon + self.args.guidance * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.vq_diffusion.module.transformer.zero_vector), dim=1)
            return log_pred
        self.vq_diffusion.module.transformer.cf_predict_start = self.vq_diffusion.module.predict_start_with_truncation(cf_predict_start, ("top"+str(self.args.truncation_rate)+'r'))
        self.vq_diffusion.module.truncation_forward = True
        self.logger.log_info(f'CF setting finished')


        if not self.model.module.learnable_cf: # empty embed
            with torch.no_grad(): # condition(CLIP) -> freeze
                condition_token = vq_diffusion.condition_codec.get_tokens([''] * self.args.batch_size) # BPE token
                cond_ = {}
                for k, v in condition_token.items():
                    v = v.to(self.device) if torch.is_tensor(v) else v
                    cond_['condition_' + k] = v
                self.empty_text_embed = vq_diffusion.transformer.condition_emb(cond_['condition_token']).float() # CLIP cond
        self.validate_epoch_fid()

    def train(self):
        start_epoch = self.last_epoch + 1
        self.start_train_time = time.time()
        self.logger.log_info('{}: global rank {}: start training...'.format(self.args.name, self.args.global_rank), check_primary=False)
        self.vq_diffusion.module.transformer.eval()
        self.model.module.transformer.train()

        batch_size = self.args.batch_size
        cf_cond_emb = self.vq_diffusion.module.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.vq_diffusion.module.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(self.vq_diffusion.module.guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.vq_diffusion.module.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.vq_diffusion.module.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
            log_new_x_recon = cf_log_x_recon + self.args.guidance * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.vq_diffusion.module.transformer.zero_vector), dim=1)
            return log_pred
            
        self.vq_diffusion.module.transformer.cf_predict_start = self.vq_diffusion.module.predict_start_with_truncation(cf_predict_start, ("top"+str(self.args.truncation_rate)+'r'))
        self.vq_diffusion.module.truncation_forward = True
        self.logger.log_info(f'CF setting finished')
        
        for epoch in range(start_epoch, self.max_epochs):
            self.train_epoch()
            self.save(force=True)
            # self.validate_epoch()