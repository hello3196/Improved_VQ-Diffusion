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

try:
    import nsml
    from nsml import IS_ON_NSML
    from nsml_utils import bind_model, Logger
except ImportError:
    nsml = None
    IS_ON_NSML = False

import wandb
from inference_VQ_Diffusion import VQ_Diffusion

from image_synthesis.utils.io import load_yaml_config

class Token_Critic(nn.Module):
    def __init__(self, config, learnable_cf = True):
        super().__init__()

        config = load_yaml_config(config)
        transformer_config = config['transformer_config']
        condition_codec_config = config['condition_codec_config']
        condition_emb_config = config['condition_emb_config']
        content_emb_config = config['content_emb_config']
        
        transformer_config['params']['content_emb_config'] = content_emb_config
        transformer_config['params']['diffusion_step'] = 100
        
        self.transformer = instantiate_from_config(transformer_config).cuda() # Token critic transformer
        self.condition_emb = instantiate_from_config(condition_emb_config).cuda() # CLIP Text embedding
        self.condition_codec = instantiate_from_config(condition_codec_config).cuda() # BPE Text tokenizer
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

    def _train_loss(self, text, input):
        """
        text condition(coco) (b,) 
        
        t           (b,)       ┐
        changed     (b, 1024)  │
        recon_token (b, 1024)  ┘ ==> input[]

        """
        self.transformer.train()
        batch_size = len(text)
        # text -> BPE tokenizing -> CLIP emb
        condition_token = self.prepare_condition(text)['condition_token'] # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.condition_emb(condition_token) # B x Ld x D   256*1024
            cond_emb = cond_emb.float() # CLIP condition
        # no text condition
        # batch['text'] = [''] * batch_size
        # cf_condition = self.prepare_condition(batch=batch)
        # cf_cond_emb = self.transformer.condition_emb(cf_condition['condition_token']).float()
        out = self.transformer(input=input['recon_token'], cond_emb=cond_emb, t=input['t']) # b, 1, 1024 logit
        out = out.squeeze() # b, 1024

        criterion = torch.nn.BCELoss()
        target = input['changed'].type(torch.float32)
        loss = criterion(out, target)

        return out, loss


if __name__ == '__main__':
    Token_Critic_model = Token_Critic(config='configs/token_critic.yaml', learnable_cf=True)
    VQ_Diffusion_model = VQ_Diffusion(config='configs/coco_tune.yaml', path='OUTPUT/pretrained_model/coco_learnable.pth')
    
    # dataset setting
    data_root = "st1/dataset/coco_vq"
    data = CocoDataset(data_root=data_root, phase='train')
    batch_size = 2

    # load from ckpt
    # checkpoint = torch.load('tc_ckpt/epoch_0_checkpoint.ckpt')
    # missing, unexpected = Token_Critic_model.load_state_dict(checkpoint["Model"], stric=False)
    # print('Model missing keys:\n', missing)
    # print('Model unexpected keys:\n', unexpected)

    optimizer = torch.optim.Adam(Token_Critic_model.transformer.parameters(), lr = 1e-4, betas = (0.9, 0.96))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9)

    os.makedirs("tc_ckpt/", exist_ok=True)
    train_loss_100 = 0
    
    for epoch in range(100):
        print(f"epoch: {epoch}")
        torch.save({'Model': Token_Critic_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,},
                    f"tc_ckpt/epoch_{epoch}_checkpoint.ckpt")
        print(f"{epoch} checkpoint is saved")

        for bb, data_i in enumerate(torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = True)):
            # data_i {"image", "text"}
            with torch.no_grad():
                vq_out = VQ_Diffusion_model.real_mask_recon(
                    data_i = data_i,
                    truncation_rate = 0.86,
                    guidance_scale = 5.0
                ) # {'t': b , 'changed': (b, 1024), 'recon_token': (b, 1024)} in cuda
            # text drop 추가 예정
            _, loss = Token_Critic_model._train_loss(data_i['text'], vq_out)
            train_loss_100 += loss.item()

            if (bb+1) % 100 == 0:
                print(f"train_loss(100 batch) {train_loss_100}")
                trin_loss_100 = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()



        

