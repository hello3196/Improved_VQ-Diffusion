# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import argparse
import os
import warnings
import time
import torch
from image_synthesis.modeling.build import build_model
from image_synthesis.data.build import build_dataloader
from image_synthesis.utils.misc import get_model_parameters_info, instantiate_from_config, seed_everything, merge_opts_to_config, modify_config_for_debug
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.engine.logger import Logger
from image_synthesis.engine.token_critic_solver import Token_Critic_Solver
from image_synthesis.distributed.launch import launch
from torch import nn
from inference_VQ_Diffusion import VQ_Diffusion
import wandb

try:
    import nsml
    from nsml import IS_ON_NSML
    # import nsml_utils.Logger as nsml_Logger
    data = os.path.join(nsml.DATASET_PATH[0], 'train')
    clip_model_path = os.path.join(nsml.DATASET_PATH[1], 'train/ViT-B-32.pt')
    diffusion_model_path = os.path.join(nsml.DATASET_PATH[2], 'train')
    diffusion_model_name = os.listdir(diffusion_model_path)[0]
    diffusion_model_path = os.path.join(diffusion_model_path, diffusion_model_name)
    vqvae_model_path = os.path.join(nsml.DATASET_PATH[3], 'train')
    vqvae_model_name = os.listdir(vqvae_model_path)[0]
    vqvae_model_path = os.path.join(vqvae_model_path, vqvae_model_name)
except ImportError:
    nsml = None
    IS_ON_NSML = False

# environment variables
NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)

class Token_Critic(nn.Module):
    def __init__(self, config, learnable_cf = False):
        super().__init__()

        transformer_config = config['transformer_config']
        condition_codec_config = config['condition_codec_config']
        condition_emb_config = config['condition_emb_config']
        content_emb_config = config['content_emb_config']
        
        transformer_config['params']['content_emb_config'] = content_emb_config
        transformer_config['params']['diffusion_step'] = 100
        
        self.learnable_cf = learnable_cf
        if self.learnable_cf:
            self.empty_text_embed = torch.nn.Parameter(torch.randn(size=(77, 512), requires_grad=False, dtype=torch.float64))
        self.transformer = instantiate_from_config(transformer_config) # Token critic transformer
        self.condition_emb = instantiate_from_config(condition_emb_config) # CLIP Text embedding
        self.condition_codec = instantiate_from_config(condition_codec_config) # BPE Text tokenizer
        self.device = "cuda"

    @torch.no_grad()
    def prepare_condition(self, text, condition=None):
        cond = text
        if torch.is_tensor(cond):
            cond = cond.to(self.device)
        cond = self.condition_codec.get_tokens(cond)
        cond_ = {}
        for k, v in cond.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cond_['condition_' + k] = v
        return cond_

    def forward(self, text, input):
        """
        text condition(coco) (b,) 
        
        t           (b,)       ┐
        changed     (b, 1024)  │
        recon_token (b, 1024)  ┘ ==> input[]

        condition_drop_rate = 0.1
        """
        self.transformer.train()
        batch_size = len(text)

        # 1) random text drop for CF-guidance -> None condition setting
        if not self.learnable_cf:
            for i in range(batch_size):
                if np.random.uniform(0, 1) < 0.1: # drop_rate 0.1
                    text[i] = ''

        # text -> BPE tokenizing -> CLIP emb
        condition_token = self.prepare_condition(text)['condition_token'] # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.condition_emb(condition_token) # B x Ld x D = b x 77 x 512
            cond_emb = cond_emb.float() # CLIP condition

        # 2) random text drop for CF-guidance -> Fine-tune from VQ-DIffusion learnable_CF
        if self.learnable_cf:
            drop = torch.rand(batch_size) < 0.1 # drop_rate 0.1
            cond_emb[drop] = self.empty_text_embed.float()

        # no text condition
        # batch['text'] = [''] * batch_size
        # cf_condition = self.prepare_condition(batch=batch)
        # cf_cond_emb = self.transformer.condition_emb(cf_condition['condition_token']).float()

        out = self.transformer(input=input['recon_token'], cond_emb=cond_emb, t=input['t']) # b, 1, 1024 logit
        out = out.squeeze() # b, 1024

        criterion = torch.nn.BCELoss()
        target = input['changed'].type(torch.float32)

        if out.dim() == 1: # batch -> odd num
            out.unsqueeze(0)
        if target.dim() == 1:
            target.unsqueeze(0)

        loss = criterion(out, target)

        return out, loss

    @torch.no_grad()
    def inference_score(self, input, cond_emb):
        self.transformer.eval()
        out = self.transformer(input=input['recon_token'], cond_emb=cond_emb, t=input['t']) # b, 1, 1024 logit
        out = out.squeeze() # b, 1024

        return out

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    parser.add_argument('--config_file', type=str, default='configs/vqvae_celeba_attribute_cond.yaml', 
                        help='path of config file')
    parser.add_argument('--name', type=str, default='', 
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file') 
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')    
    parser.add_argument('--log_frequency', type=int, default=100, 
                        help='print frequency (default: 100)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='path to model that need to be loaded, '
                             'used for loading pretrained model')
    parser.add_argument('--resume_name', type=str, default=None,
                        help='resume one experiment with the given name')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatically resume the training')

    # args for ddp
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default=DIST_URL, 
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    parser.add_argument('--sync_bn', action='store_true', 
                        help='use sync BN layer')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')
    parser.add_argument('--timestamp', action='store_true', # default=True,
                        help='use tensorboard for logging')
    # args for random
    parser.add_argument('--seed', type=int, default=None, 
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', action='store_true', 
                        help='set cudnn.deterministic True')

    parser.add_argument('--amp', action='store_true', # default=True,
                        help='automatic mixture of precesion')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='set as debug mode')

    parser.add_argument('--only_val', type=bool, default=False,
                        help='measure metric w/o training')

    parser.add_argument('--use_my_ckpt', type=bool, default=True,
                        help='whether to use our ckpt')

    # args for experiment setting
    parser.add_argument('--batch_size', type=int, default=4, 
                    help='batch_size (default: 4)')
    parser.add_argument('--step', type=int, default=16, 
                        help='decoding step (available: 16(default), 50, 100)')
    parser.add_argument('--guidance', type=float, default=5, 
                        help='CF-guidance scale (default: 5)')
    parser.add_argument('--schedule', type=int, default=5, 
                        help='5:uniform, 6:purity, 7:revoke (only when step == 16, 1~4 available)')
    parser.add_argument('--truncation_rate', type=float, default=0.86, 
                        help='truncation rate (default: 0.86)')

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )  

    args = parser.parse_args()
    print(args)
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    if args.resume_name is not None:
        args.name = args.resume_name
        args.config_file = os.path.join(args.output, args.resume_name, 'configs', 'config.yaml')
        args.auto_resume = True
    else:
        if args.name == '':
            args.name = os.path.basename(args.config_file).replace('.yaml', '')
        if args.timestamp:
            assert not args.auto_resume, "for timstamp, auto resume is hard to find the save directory"
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            args.name = time_str + '-' + args.name

    # modify args for debugging
    if args.debug:
        args.name = 'debug'
        if args.gpu is None:
            args.gpu = 0

    args.save_dir = os.path.join(args.output, args.name)
    return args

def main():
    args = get_args()

    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    launch(main_worker, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))


def main_worker(local_rank, args):

    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    # load config
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    if args.debug:
        config = modify_config_for_debug(config)

    # get logger
    logger = Logger(args)
    logger.save_config(config)

    # get model 
    VQ_Diffusion_model = VQ_Diffusion(config='configs/coco_tune.yaml', path=diffusion_model_path)
    Token_Critic_model = Token_Critic(config=config, learnable_cf=True).cuda()
    # wandb.init(project='TC train', name = 'layer12_scratch_coco_train')
    # print(model)
    if args.sync_bn:
        VQ_Diffusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(VQ_Diffusion_model)
        Token_Critic_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Token_Critic_model)

    # get dataloader
    config['dataloader']['batch_size'] = args.batch_size
    dataloader_info = build_dataloader(config, args)

    # get solver
    solver = Token_Critic_Solver(config=config, args=args, token_critic_model=Token_Critic_model,
                                diffusion_model=VQ_Diffusion_model, dataloader=dataloader_info, logger=logger)

    # resume 
    if args.load_path is not None: # only load the model parameters
        solver.resume(path=args.load_path,
                      # load_model=True,
                      load_optimizer_and_scheduler=False,
                      load_others=False)
    if args.auto_resume:
        solver.resume()
    # with torch.autograd.set_detect_anomaly(True):
    #     solver.train()

    # CF guidance setting
    batch_size = args.batch_size
    cf_cond_emb = VQ_Diffusion_model.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
    def cf_predict_start(log_x_t, cond_emb, t):
        log_x_recon = VQ_Diffusion_model.model.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
        if abs(VQ_Diffusion_model.model.guidance_scale - 1) < 1e-3:
            return torch.cat((log_x_recon, VQ_Diffusion_model.model.transformer.zero_vector), dim=1)
        cf_log_x_recon = VQ_Diffusion_model.model.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
        log_new_x_recon = cf_log_x_recon + VQ_Diffusion_model.model.guidance_scale * (log_x_recon - cf_log_x_recon)
        log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
        log_new_x_recon = log_new_x_recon.clamp(-70, 0)
        log_pred = torch.cat((log_new_x_recon, VQ_Diffusion_model.model.transformer.zero_vector), dim=1)
        return log_pred
    VQ_Diffusion_model.model.transformer.cf_predict_start = VQ_Diffusion_model.model.predict_start_with_truncation(cf_predict_start, ("top"+str(args.truncation_rate)+'r'))
    VQ_Diffusion_model.model.truncation_forward = True

    if args.only_val:
        solver.validate()
    else:
        solver.train()

if __name__ == '__main__':
    main()
