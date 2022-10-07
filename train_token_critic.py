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

from image_synthesis.utils.io import load_yaml_config

class Token_Critic(nn.Module):
    def __init__(self, config, learnable_cf = False):
        super().__init__()


        config = load_yaml_config(config)
        transformer_config = config['transformer_config']
        condition_codec_config = config['condition_codec_config']
        condition_emb_config = config['condition_emb_config']
        content_emb_config = config['content_emb_config']
        
        transformer_config['params']['content_emb_config'] = content_emb_config
        transformer_config['params']['diffusion_step'] = 100
        

        self.learnable_cf = learnable_cf
        if self.learnable_cf:
            self.empty_text_embed = torch.nn.Parameter(torch.randn(size=(77, 512), requires_grad=False, dtype=torch.float64)).cuda()

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

if __name__ == '__main__':
    
    Token_Critic_model = Token_Critic(config='configs/token_critic.yaml', learnable_cf=True)
    VQ_Diffusion_model = VQ_Diffusion(config='configs/coco_tune.yaml', path='OUTPUT/pretrained_model/coco_learnable.pth')
    wandb.init(project='TC train', name = 'layer12_scratch_coco_train')
    # dataset setting
    data_root = "/st1/dataset/coco_vq"
    data = CocoDataset(data_root=data_root, phase='train')
    batch_size = 8

    # load from ckpt
    # checkpoint = torch.load('tc_ckpt/epoch_0_checkpoint.ckpt')
    # missing, unexpected = Token_Critic_model.load_state_dict(checkpoint["Model"], stric=False)
    # print('Model missing keys:\n', missing)
    # print('Model unexpected keys:\n', unexpected)

    # fine tune from VQ-diffusion weights
    # weights = torch.load('OUTPUT/pretrained_model/coco_learnable.pth')['ema']
    # # remove last classifier layer for load (same name, but not used)
    # weights.pop('transformer.to_logits.1.weight', None)
    # weights.pop('transformer.to_logits.1.bias', None)
    # missing, _ = Token_Critic_model.load_state_dict(weights, strict=False)
    # print("Finetune from VQ-Diffusion Model")
    # print("Missing weights:\n", missing)


    optimizer = torch.optim.Adam(Token_Critic_model.transformer.parameters(), lr = 3.0e-6, betas = (0.9, 0.96))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.5)
    scheduler = ReduceLROnPlateauWithWarmup(optimizer, factor=0.5, patience=60000, min_lr=1.0e-6, threshold=1.0e-1,
        threshold_mode='rel', warmup_lr=4.5e-4, warmup=5000)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = True, drop_last = True)
    os.makedirs("tc_ckpt/", exist_ok=True)
    
    torch.save({'Model': Token_Critic_model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': 0,},
                    f"tc_ckpt/init_checkpoint_scratch_layer_12.ckpt")

    print("Training start")
    for epoch in range(100):
        print(f"epoch: {epoch}")
        train_loss = 0
        train_loss_1000 = 0
        for bb, data_i in enumerate(data_loader):

            # data_i {"image", "text"}
            with torch.no_grad():
                vq_out = VQ_Diffusion_model.real_mask_recon(
                    data_i = data_i,
                    truncation_rate = 0.86,
                    guidance_scale = 5.0
                ) # {'t': b , 'changed': (b, 1024), 'recon_token': (b, 1024)} in cuda
            # text drop 추가 예정
            _, loss = Token_Critic_model._train_loss(data_i['text'], vq_out)

            train_loss += loss.item()
            train_loss_1000 += loss.item()
            if (bb+1) % 1000 == 0:
                print(f"train_loss_1000: {train_loss_1000 / 1000}")
                wandb.log({'train_loss_1000': train_loss_1000 / 1000})
                train_loss_1000 = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # if (bb+1) % 1000 == 0:
            #     torch.save({'Model': Token_Critic_model.state_dict(),
            #             # 'optimizer_state_dict': optimizer.state_dict(),
            #             'epoch': epoch,},
            #             f"tc_ckpt/epoch_{epoch}_iter_{bb}_checkpoint.ckpt")
        
        
        print(f"{epoch}_train_loss: {train_loss / len(data_loader)}")
        wandb.log({'epoch_train_loss' : train_loss / len(data_loader)})
        # wandb log
        # wandb.log({'train_loss' : train_loss / len(data_loader)})

        torch.save({'Model': Token_Critic_model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,},
                    f"tc_ckpt/epoch_{epoch}_checkpoint_scratch_layer_12.ckpt")
        print(f"{epoch} checkpoint is saved")




        

