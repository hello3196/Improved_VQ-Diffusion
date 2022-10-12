import os
import visdom
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info
import torch

try:
    import nsml
    from nsml import IS_ON_NSML
except ImportError:
    nsml = None
    IS_ON_NSML = False

class Logger(object):
    def __init__(self,):
        self.last = None
        self.viz = nsml.Visdom(visdom=visdom)
    
    def scalar_summary(self, tag, value, step):
        if self.last and self.last['step'] != step:
            nsml.report(**self.last)
            self.last = None
        if self.last is None:
            self.last = {
                'step': step, 
                'iter': step, 
                'epoch': 1
            }
        self.last[tag] = value

    def images_summary(self, tag, images, step=0):
        self.viz.image(
            images, 
            opts = dict(title=tag, caption='%s/%d' % (tag, step)))

    def histo_summary(self, tag, values, step, bins=1000):
        pass


def bind_model(last_epoch, last_iter, model, ema, clip_grad_norm, optimizer_and_scheduler, local_rank, load_others, load_optimizer_and_scheduler):
    def load(filename, **kwargs):
        load_path = os.path.join(filename, 'model.pth')
        print("loading from ", load_path)
        state_dict = torch.load(load_path, map_location='cuda:{}'.format(local_rank))
        if load_others:
            last_epoch = state_dict['last_epoch']
            last_iter = state_dict['last_iter']
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            try:
                model.module.load_state_dict(state_dict['model'])
            except:
                model_dict = model.module.state_dict()
                temp_state_dict = {k:v for k,v in state_dict['model'].items() if k in model_dict.keys()}
                model_dict.update(temp_state_dict)
                model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(state_dict['model'])

        if 'ema' in state_dict and ema is not None:
            try:
                ema.load_state_dict(state_dict['ema'])
            except:
                model_dict = ema.state_dict()
                temp_state_dict = {k:v for k,v in state_dict['ema'].items() if k in model_dict.keys()}
                model_dict.update(temp_state_dict)
                ema.load_state_dict(model_dict)

        if 'clip_grad_norm' in state_dict and clip_grad_norm is not None and state_dict['clip_grad_norm'] is not None:
            clip_grad_norm.load_state_dict(state_dict['clip_grad_norm'])
        
        if state_dict['optimizer_and_scheduler'] is not None:
            for op_sc_n, op_sc in state_dict['optimizer_and_scheduler'].items():
                for k in op_sc:
                    if k in ['optimizer', 'scheduler']:
                        for kk in op_sc[k]:
                            if kk == 'module' and load_optimizer_and_scheduler:
                                optimizer_and_scheduler[op_sc_n][k][kk].load_state_dict(op_sc[k][kk])
                            elif load_others: # such as step_iteration, ...
                                optimizer_and_scheduler[op_sc_n][k][kk] = op_sc[k][kk]
                    elif load_others: # such as start_epoch, end_epoch, ....
                        optimizer_and_scheduler[op_sc_n][k] = op_sc[k]

    def save(filename, **kwargs):
        save_path = os.path.join(filename, 'model.pth')
        print("saving at ", save_path)
        state_dict = {
                    'last_epoch': last_epoch,
                    'last_iter': last_iter,
                    'model': model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict() 
                }
        if ema is not None:
            state_dict['ema'] = ema.state_dict()
        if clip_grad_norm is not None:
            state_dict['clip_grad_norm'] = clip_grad_norm.state_dict()

        op_and_sc = {}
        for op_sc_n, op_sc in optimizer_and_scheduler.items():
            state_ = {}
            for k in op_sc:
                if k in ['optimizer', 'scheduler']:
                    op_or_sc = {kk: vv for kk, vv in op_sc[k].items() if kk != 'module'}
                    op_or_sc['module'] = op_sc[k]['module'].state_dict()
                    state_[k] = op_or_sc
                else:
                    state_[k] = op_sc[k]
            op_and_sc[op_sc_n] = state_

        state_dict['optimizer_and_scheduler'] = op_and_sc

        torch.save(state_dict, save_path)

    if IS_ON_NSML is True:
        nsml.bind(save=save, load=load)