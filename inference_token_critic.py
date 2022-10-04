import os
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
from torch import nn
import torch.nn.functional as F
# import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image


import scipy.linalg
import get_FID

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.utils.misc import get_model_parameters_info
import image_synthesis.modeling.modules.clip.clip as clip
from image_synthesis.data.mscoco_dataset import CocoDataset 
from image_synthesis.engine.lr_scheduler import ReduceLROnPlateauWithWarmup
try:
    import nsml
    from nsml import IS_ON_NSML
    from nsml_utils import bind_model, Logger
except ImportError:
    nsml = None
    IS_ON_NSML = False

import wandb
from inference_VQ_Diffusion import VQ_Diffusion
from train_token_critic import Token_Critic
from image_synthesis.utils.io import load_yaml_config

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

class VQ_Critic(nn.Module):
    def __init__(self, vq='coco', tc_config='configs/token_critic.yaml', tc_learnable_cf=True):
        super().__init__()
        if vq == 'coco':
            self.VQ_Diffusion = VQ_Diffusion(config='configs/coco_tune.yaml', path='OUTPUT/pretrained_model/coco_learnable.pth')
        elif vq == 'ithq':
            self.VQ_Diffusion = VQ_Diffusion(config='configs/ithq.yaml', path='OUTPUT/pretrained_model/ithq_learnable.pth')
        
        self.Token_Critic = Token_Critic(config=tc_config, learnable_cf=tc_learnable_cf)
        self.tc_learnable = tc_learnable_cf
        self.device = "cuda"

    def load_tc(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        checkpoint["Model"]
        print('Token Critic Load from', ckpt_path)
        missing, unexpected = self.Token_Critic.load_state_dict(checkpoint["Model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)  

    def n_t(self, t):
        # t smaller(initial stage) -> variance bigger
        a = 0.0
        b = 0.0
        p = t / 100 * a + b
        return p

    @torch.no_grad()
    def prepare_condition(self, text):
        cond = text
        if torch.is_tensor(cond):
            cond = cond.to(self.device)
        cond = self.Token_Critic.condition_codec.get_tokens(cond)
        cond_ = {}
        for k, v in cond.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cond_['condition_' + k] = v
        return cond_

    @torch.no_grad()
    def inference_generate_sample_with_condition(self, text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16):
        """
        step: 16, 50, 100

        prepare condition: text -> tokenize -> CLIP -> cond_emb
        initialize: All mask
        
        iterate decoding step:
            1) VQ: 1 step reconstruction
            2) TC: Masking based on score
        """
        save_root = 'RESULT'
        os.makedirs(save_root, exist_ok=True)
        str_cond = str(text) + '_' + str(step)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        # masking list & Diffusion time step
        if step == 16:
            n_sample = [64] * 16
            time_list = [index for index in range(100 -5, -1, -6)]
        elif step == 50:
            n_sample = [index for index in range(100 -1, -1, -2)]
        else: # 100
            n_sample = [index for index in range(100 -1, -1, -1)]

        n_mask = []
        for s in range(1, step):
            n_sample[s] += n_sample[s-1]
        for s in range(step):
            n_mask.append(1024 - n_sample[s]) # the number of masking tokens each step


        # setting for VQ_Diffusion
        self.VQ_Diffusion.model.guidance_scale = vq_guidance
        self.VQ_Diffusion.model.learnable_cf = self.VQ_Diffusion.model.transformer.learnable_cf = True

        cf_cond_emb = self.VQ_Diffusion.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.VQ_Diffusion.model.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(self.VQ_Diffusion.model.guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.VQ_Diffusion.model.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.VQ_Diffusion.model.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
     
            log_new_x_recon = cf_log_x_recon + self.VQ_Diffusion.model.guidance_scale * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.VQ_Diffusion.model.transformer.zero_vector), dim=1)
            return log_pred

        sample_type = "top"+str(vq_tr)+'r'
        self.VQ_Diffusion.model.transformer.cf_predict_start = self.VQ_Diffusion.model.predict_start_with_truncation(cf_predict_start, sample_type.split(',')[0])
        self.VQ_Diffusion.model.truncation_forward = True

        # prepare condition: text -> tokenize -> CLIP -> cond_emb
        text_list = [text] * batch_size
        condition_token = self.prepare_condition(text_list) # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
            cond_emb = cond_emb.float() # CLIP condition

        # initialize: All mask 
        device = self.device
        zero_logits = torch.zeros((batch_size, self.VQ_Diffusion.model.transformer.num_classes-1, self.VQ_Diffusion.model.transformer.shape),device=device)
        one_logits = torch.ones((batch_size, 1, self.VQ_Diffusion.model.transformer.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        # iterate decoding step
        with torch.no_grad():
            for i, (n, diffusion_index) in enumerate(zip(n_sample[:-1], time_list[:-1])): # before last step
                # 1) VQ: 1 step reconstruction
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024
                out2_idx = torch.full_like(out_idx, self.VQ_Diffusion.model.transformer.num_classes-1).to(out_idx.device) # all mask index list

                # 2) TC: Masking based on score
                t_1 = torch.full((batch_size,), time_list[i+1], device=device, dtype=torch.long) # t-1 step
                score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024

                # score += self.n_t(diffusion_index) # n(t) for randomness
                # score *= 2 # weight sampling

                score = 1 - score # (1 - score) = unchanged confidence score (because score means changed confidence)
                for ii in range(batch_size):
                    sel = torch.multinomial(score[ii], n)
                    out2_idx[ii][sel] = out_idx[ii][sel]
                log_z = index_to_log_onehot(out2_idx, self.VQ_Diffusion.model.transformer.num_classes)
            
            # Final step
            t = torch.full((batch_size,), time_list[-1], device=device, dtype=torch.long)
            _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
            out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
            content_token = log_onehot_to_index(out) # b, 1024
        
        # decoding token
        content = self.VQ_Diffusion.model.content_codec.decode(content_token)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)




if __name__ == '__main__':
    model = VQ_Critic(vq='coco', tc_config='configs/token_critic.yaml', tc_learnable_cf=True)
    # model.load_tc('tc_ckpt/epoch_0_checkpoint.ckpt')

    model.inference_generate_sample_with_condition(text="teddy bear playing in the pool", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=64)
    
