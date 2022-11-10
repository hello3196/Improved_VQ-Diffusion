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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import scipy.linalg
import get_FID
from tqdm.auto import tqdm

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
        self.device = "cuda"
        self.num_classes = self.VQ_Diffusion.model.transformer.num_classes

    def load_tc(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        # checkpoint["Model"]
        print('Token Critic Load from', ckpt_path)
        missing, unexpected = self.Token_Critic.load_state_dict(checkpoint["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)  
        self.tc_learnable = self.Token_Critic.learnable_cf
        print(f"tc_learnable: {self.tc_learnable}")
        self.Token_Critic.empty_text_embed = torch.load('cf_emb.pth').cuda()
        print("cf_emb load")

        self.Token_Critic.eval()
        

    def n_t(self, t, a=0.1, b=0.2):
        # t smaller(initial stage) -> variance bigger
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

    def score_matrix_log(self, score): # score (b, 1024)
        for i in range(score.shape[0]):
            score_2d = score[i].view(32, 32)
            df = pd.DataFrame(score_2d.to('cpu').numpy())
            fig, ax = plt.subplots(figsize = (20, 20))
            sns.heatmap(df, annot=df, fmt='.2f', cbar=False, vmin=0, vmax=1, cmap="Greys")
            wandb.log({f"{i}_score_matrix": wandb.Image(fig)})
            plt.close()

    def score_matrix_log_tc(self, score): # score (b, 1024)
        for i in range(score.shape[0]):
            score_2d = score[i].view(32, 32)
            df = pd.DataFrame(score_2d.to('cpu').numpy())
            fig, ax = plt.subplots(figsize = (20, 20))
            sns.heatmap(df, annot=df, fmt='.2f', cbar=False, vmin=0, vmax=1, cmap="Greys")
            wandb.log({f"{i}_score_matrix_tc": wandb.Image(fig)})
            plt.close()

    def recon_image_with_attn_log(self, out_idx, score): # out_idx (b, 1024)
        content = self.VQ_Diffusion.model.content_codec.decode(out_idx)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            # im = Image.fromarray(content[b])
            score_2d = score[b].view(32, 32).to('cpu').numpy()
            score_2d = cv2.resize(score_2d, (256, 256)) # interpolate for 256x256
            cam = content[b] / 255. + cv2.applyColorMap(np.uint8(255 * score_2d), cv2.COLORMAP_JET) / 255.
            cam = np.uint8(cam / np.max(cam) * 255)
            im = Image.fromarray(cam)
            wandb.log({f"{b:02d}_score_attn" : wandb.Image(im)})

    def recon_image_log(self, out_idx): # out_idx (b, 1024)
        content = self.VQ_Diffusion.model.content_codec.decode(out_idx)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            wandb.log({f"{b:02d}_step recon" : wandb.Image(im)})

    def recon_image_log_tc(self, out_idx): # out_idx (b, 1024)
        content = self.VQ_Diffusion.model.content_codec.decode(out_idx)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            wandb.log({f"{b:02d}_step recon_tc" : wandb.Image(im)})

    def mask_token_log(self, out_idx): # out2_idx (b, 1024 with mask token)
        for i in range(out_idx.shape[0]):
            out_idx_2d = out_idx[i].view(32, 32)
            df = pd.DataFrame(out_idx_2d.to('cpu').numpy())
            an = df.replace(self.num_classes - 1, 'mask').astype('str')
            df = df.replace(self.num_classes - 1, -(self.num_classes - 1)) # for cmap
            fig, ax = plt.subplots(figsize = (20, 20))
            sns.heatmap(df, annot=an, cbar=False, fmt = '', vmin = -(self.num_classes - 1), vmax = self.num_classes - 1)
            wandb.log({f"{i}_content": wandb.Image(fig)})   
            plt.close() 

    def mask_token_log_tc(self, out_idx): # out2_idx (b, 1024 with mask token)
        for i in range(out_idx.shape[0]):
            out_idx_2d = out_idx[i].view(32, 32)
            df = pd.DataFrame(out_idx_2d.to('cpu').numpy())
            an = df.replace(self.num_classes - 1, 'mask').astype('str')
            df = df.replace(self.num_classes - 1, -(self.num_classes - 1)) # for cmap
            fig, ax = plt.subplots(figsize = (20, 20))
            sns.heatmap(df, annot=an, cbar=False, fmt = '', vmin = -(self.num_classes - 1), vmax = self.num_classes - 1)
            wandb.log({f"{i}_content_tc": wandb.Image(fig)})   
            plt.close()

    @torch.no_grad()
    def accuracy_test(self, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None):

        data_root = "st1/dataset/coco_vq"
        data = CocoDataset(data_root=data_root, phase='val')
        batch_size = 4

        # setting for VQ_Diffusion
        self.VQ_Diffusion.model.guidance_scale = vq_guidance
        self.VQ_Diffusion.model.learnable_cf = self.VQ_Diffusion.model.transformer.learnable_cf = True

        cf_cond_emb = self.VQ_Diffusion.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.tc_learnable: 
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1).float()
        else:
            none_text = [''] * batch_size
            condition_token = self.prepare_condition(none_text) # BPE token
            with torch.no_grad(): # condition(CLIP) -> freeze
                tc_cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
                tc_cf_cond_emb = tc_cond_emb.float() # CLIP condition

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

        if self.tc_learnable: 
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1).float()
        else:
            none_text = [''] * batch_size
            condition_token = self.prepare_condition(none_text) # BPE token
            with torch.no_grad(): # condition(CLIP) -> freeze
                tc_cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
                tc_cf_cond_emb = tc_cond_emb.float() # CLIP condition

        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = True)
        print('Validation start')
        for bb, data_i in enumerate(data_loader):
            with torch.no_grad():
                vq_out = self.VQ_Diffusion.real_mask_recon(
                    data_i = data_i,
                    truncation_rate = 0.86,
                    guidance_scale = 5.0
                ) # {'t': b , 'changed': (b, 1024), 'recon_token': (b, 1024)} in cuda
                condition_token = self.Token_Critic.prepare_condition(data_i['text'])['condition_token'] # BPE token
                with torch.no_grad(): # condition(CLIP) -> freeze
                    cond_emb = self.Token_Critic.condition_emb(condition_token) # B x Ld x D   256*1024
                    cond_emb = cond_emb.float() # CLIP condition
                label = vq_out['changed'] # b, 1024
                time =  vq_out['t']
                # score = self.Token_Critic.transformer(input={'t': time, 'recon_token': vq_out['recon_token']}, cond_emb=cond_emb) # b, 1024
                score = self.Token_Critic.transformer(input=vq_out['recon_token'], cond_emb=cond_emb, t=time).squeeze() # b, 1024
                
                gt_token = self.VQ_Diffusion.model.content_codec.get_tokens(data_i['image'].cuda())['token'].cuda()
                gt_image = self.VQ_Diffusion.model.content_codec.decode(gt_token).permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                recon_image = self.VQ_Diffusion.model.content_codec.decode(vq_out['recon_token']).permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                for i in range(batch_size):
                    tt = label[i].nonzero().squeeze() # changed index list
                    num_changed = len(tt)
                    _, ss = torch.topk(score[i], num_changed, dim=0)  # expected changed index list
                    inter = len(set(tt.to('cpu').numpy()).intersection(set(ss.to('cpu').numpy())))
                    acc = inter / (2 * num_changed - inter +1e-6)
                    loss = torch.nn.BCELoss(reduction='none')(score[i].type(torch.float32), label[i].type(torch.float32))
                    print(f"Time: {time[i]} / Acc: {acc:.2f} / loss: {loss.mean():.3f} / num_changed: {num_changed}")
                    # changed map
                    changed_df = pd.DataFrame(label[i].view(32, 32).to('cpu').numpy()) 
                    changed_fig, ax = plt.subplots(figsize = (20, 20))
                    sns.heatmap(changed_df, cbar=False, cmap='Greys')
                    # expected_score map
                    score_df = pd.DataFrame(score[i].view(32, 32).to('cpu').numpy())
                    score_fig, ax = plt.subplots(figsize = (20, 20))
                    sns.heatmap(score_df, annot=score_df, fmt='.2f', cbar=False, vmin=0, vmax=1, cmap='Greys')
                    # loss location map
                    loss_df = pd.DataFrame(loss.view(32, 32).to('cpu').numpy())
                    loss_fig, ax = plt.subplots(figsize = (20, 20))
                    sns.heatmap(loss_df, annot=loss_df, fmt='.2f', cbar=False, vmin=0, vmax=1, cmap='Greys')

                    wandb.log({"recon_image": wandb.Image(Image.fromarray(recon_image[i]), caption= f"Time({time[i].item()})_Acc({acc:.2f})_Loss_({loss.mean():.2f})_{data_i['text'][i]}"),
                        "gt image": wandb.Image(Image.fromarray(gt_image[i]), caption=f"{data_i['text'][i]}"),
                        "changed": wandb.Image(changed_fig),
                        "expected_score": wandb.Image(score_fig),
                        "loss_location": wandb.Image(loss_fig),
                        "Time": time[i],
                        "Acc": acc,
                        "loss": loss.mean(),
                        "num_changed": num_changed
                     })
                    plt.close()
        

    @torch.no_grad()
    def inference_generate_sample_with_condition(self, text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log="s,t,r", a=0., b=0., score_weight = 1., token_critic=True, gaussian_alpha = 1.):
        """
        step: 16, 50, 100

        prepare condition: text -> tokenize -> CLIP -> cond_emb
        initialize: All mask
        
        iterate decoding step:
            1) VQ: 1 step reconstruction
            2) TC: Masking based on score
        """
        log_set = wandb_log.split(',')
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
            n_sample = [10] + [21, 20] * 24 + [30]
            time_list = [index for index in range(100 -1, -1, -2)]
        else: # 100
            n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
            time_list = [index for index in range(100 -1, -1, -1)]

        n_mask = []
        for s in range(1, step):
            n_sample[s] += n_sample[s-1]
        for s in range(step):
            n_mask.append(1024 - n_sample[s]) # the number of masking tokens each step

        
        # setting for VQ_Diffusion
        self.VQ_Diffusion.model.guidance_scale = vq_guidance
        self.VQ_Diffusion.model.learnable_cf = self.VQ_Diffusion.model.transformer.learnable_cf = True

        cf_cond_emb = self.VQ_Diffusion.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.tc_learnable: 
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1).float()
        else:
            none_text = [''] * batch_size
            condition_token = self.prepare_condition(none_text) # BPE token
            with torch.no_grad(): # condition(CLIP) -> freeze
                tc_cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
                tc_cf_cond_emb = tc_cond_emb.float() # CLIP condition

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
        self.VQ_Diffusion.model.truncation_forward = False

        # prepare condition: text -> tokenize -> CLIP -> cond_emb
        text_list = [text] * batch_size
        condition_token = self.prepare_condition(text_list) # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
            cond_emb = cond_emb.float() # CLIP condition

        # initialize: All mask 
        device = self.device
        zero_logits = torch.zeros((batch_size, self.num_classes-1, self.VQ_Diffusion.model.transformer.shape),device=device)
        one_logits = torch.ones((batch_size, 1, self.VQ_Diffusion.model.transformer.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        # iterate decoding step
        self.token_critic_on = token_critic

        with torch.no_grad():
            for i, (n, diffusion_index) in tqdm(enumerate(zip(n_sample[:-1], time_list[:-1]))): # before last step
                # schedule test => 앞부분 tc, 뒷부분 random
                # if i % 2 == 0:
                #     self.token_critic_on = True
                # else:
                #     self.token_critic_on = False

                # 1) VQ: 1 step reconstruction
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024
                out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list

                # 2) TC: Masking based on score
                t_1 = torch.full((batch_size,), time_list[i+1], device=device, dtype=torch.long) # t-1 step

                if self.token_critic_on == True:
                    print(f"Token_Critic step i={i}")
                    score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                    if tc_guidance != None:
                        cf_score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=tc_cf_cond_emb)
                        score = cf_score + tc_guidance * (score - cf_score)
                    score = 1 - score # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                    score = score.clamp(min = 0) # score clamp for CF minus probability
                    # score = score * score_weight
                    score = (score - score.mean(1, keepdim=True)) * score_weight + score.mean(1, keepdim=True)
                    score = score.clamp(min = 0)
                    score += self.n_t(diffusion_index, a, b) # n(t) for randomness
                    score += gaussian_alpha * diffusion_index / 100 * torch.rand_like(score).to(out_idx.device)
                    # score *= 2 # weight sampling

                else: # token_critic False
                    print(f"Random step i={i}")
                    score = torch.ones_like(out_idx).float().cuda()


                for ii in range(batch_size):
                    # sel = torch.multinomial(score[ii], n)
                    _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                    out2_idx[ii][sel] = out_idx[ii][sel]
                log_z = index_to_log_onehot(out2_idx, self.num_classes)

                # log
                if 's' in log_set:
                    self.score_matrix_log(score)
                if 'r' in log_set:
                    self.recon_image_log(out_idx)
                if 't' in log_set:
                    self.mask_token_log(out2_idx)

            # Final step
            t = torch.full((batch_size,), time_list[-1], device=device, dtype=torch.long)
            _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
            out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
            content_token = log_onehot_to_index(out) # b, 1024

            if 't' in log_set:
                self.mask_token_log(content_token)
        
        # decoding token
        content = self.VQ_Diffusion.model.content_codec.decode(content_token)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result" : wandb.Image(im)})


    @torch.no_grad()
    def refine_test(self, text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, start_step=8, wandb_log="s,t,r", a=0., b=0., score_weight=1., purity=True):
        """
        step: 16, 50, 100

        prepare condition: text -> tokenize -> CLIP -> cond_emb
        initialize: All mask
        
        iterate decoding step:
            1) VQ: 1 step reconstruction
            2) TC: Masking based on score

        purity
            True -> purity sampling
            False -> random revoke
        """
        log_set = wandb_log.split(',')
        save_root = 'RESULT'
        os.makedirs(save_root, exist_ok=True)
        str_cond = str(text) + '_' + str(step)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        # masking list & Diffusion time step
        self.purity = purity

        if step == 16:
            n_sample = [64] * 16
            time_list = [index for index in range(100 -5, -1, -6)]
        elif step == 50:
            n_sample = [10] + [21, 20] * 24 + [30]
            time_list = [index for index in range(100 -1, -1, -2)]
        else: # 100
            n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
            time_list = [index for index in range(100 -1, -1, -1)]

        purity_mask = n_sample
        n_mask = []
        for s in range(1, step):
            n_sample[s] += n_sample[s-1]
        for s in range(step):
            n_mask.append(1024 - n_sample[s]) # the number of masking tokens each step

        
        # setting for VQ_Diffusion
        self.VQ_Diffusion.model.guidance_scale = vq_guidance
        self.VQ_Diffusion.model.learnable_cf = self.VQ_Diffusion.model.transformer.learnable_cf = True

        cf_cond_emb = self.VQ_Diffusion.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.tc_learnable: 
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1).float()
        else:
            none_text = [''] * batch_size
            condition_token = self.prepare_condition(none_text) # BPE token
            with torch.no_grad(): # condition(CLIP) -> freeze
                tc_cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
                tc_cf_cond_emb = tc_cond_emb.float() # CLIP condition

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
        self.VQ_Diffusion.model.truncation_forward = False

        # prepare condition: text -> tokenize -> CLIP -> cond_emb
        text_list = [text] * batch_size
        condition_token = self.prepare_condition(text_list) # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
            cond_emb = cond_emb.float() # CLIP condition

        # initialize: All mask 
        device = self.device
        zero_logits = torch.zeros((batch_size, self.num_classes-1, self.VQ_Diffusion.model.transformer.shape),device=device)
        one_logits = torch.ones((batch_size, 1, self.VQ_Diffusion.model.transformer.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        with torch.no_grad():
            for i, (n, diffusion_index) in tqdm(enumerate(zip(n_sample[:-1], time_list[:-1]))): # before last step
                if i < start_step: # random step
                    # 1) VQ: 1 step reconstruction
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                    out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                    out_idx = log_onehot_to_index(out) # b, 1024

                    if self.purity:
                        log_z_idx = log_onehot_to_index(log_z)
                        out2_idx = log_z_idx.clone() # previous
                        score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                        score /= (score.max(dim=1, keepdim=True).values + 1e-10)
                    else: # random revoke
                        out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list
                        score = torch.ones_like(out_idx).float().cuda()
                    
                    if i != start_step - 1:
                        if self.purity:
                            print(f"Purity initialize i = {i}")
                            for ii in range(batch_size):
                                sel = torch.multinomial(score[ii], purity_mask[i])
                                # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                                out2_idx[ii][sel] = out_idx[ii][sel]
                            log_z = index_to_log_onehot(out2_idx, self.num_classes)
                        else:
                            print(f"Random initialize i = {i}")
                            for ii in range(batch_size):
                                sel = torch.multinomial(score[ii], n)
                                # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                                out2_idx[ii][sel] = out_idx[ii][sel]
                            log_z = index_to_log_onehot(out2_idx, self.num_classes)

                    else: # i == start_step - 1
                        t_1 = torch.full((batch_size,), time_list[i+1], device=device, dtype=torch.long) # t-1 step
                        print(f"Token Critic start i = {i}")
                        score_tc = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                        if tc_guidance != None:
                            cf_score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=tc_cf_cond_emb)
                            score_tc = cf_score + tc_guidance * (score_tc - cf_score)
                        score_tc = 1 - score_tc # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                        score_tc = score_tc.clamp(min = 0) # score clamp for CF minus probability
                        # score_tc = score_tc * score_weight
                        score_tc = (score_tc - score_tc.mean(1, keepdim=True)) * score_weight + score_tc.mean(1, keepdim=True)
                        score_tc = score_tc.clamp(min = 0)
                        score_tc += self.n_t(diffusion_index, a, b) # n(t) for randomness

                        if self.purity:
                            for ii in range(batch_size):
                                sel = torch.multinomial(score[ii], purity_mask[i])
                                # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                                out2_idx[ii][sel] = out_idx[ii][sel]
                            log_z = index_to_log_onehot(out2_idx, self.num_classes)
                        else:
                            for ii in range(batch_size):
                                sel = torch.multinomial(score[ii], n)
                                # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                                out2_idx[ii][sel] = out_idx[ii][sel]
                            log_z = index_to_log_onehot(out2_idx, self.num_classes)

                        out2_idx_tc = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device)
                        for ii in range(batch_size):
                            sel = torch.multinomial(score_tc[ii], n)
                            # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                            out2_idx_tc[ii][sel] = out_idx[ii][sel]
                        log_z_tc = index_to_log_onehot(out2_idx_tc, self.num_classes)

                        if 'r' in log_set:
                            self.recon_image_log(out_idx)
                            self.recon_image_log_tc(out_idx)

                        if 't' in log_set:
                            self.mask_token_log(out2_idx)
                            self.mask_token_log_tc(out2_idx_tc)
     
                else: # random & tc
                    # 1) VQ: 1 step reconstruction
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                    out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                    out_idx = log_onehot_to_index(out) # b, 1024

                    if self.purity:
                        log_z_idx = log_onehot_to_index(log_z)
                        out2_idx = log_z_idx.clone() # previous
                        score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                        score /= (score.max(dim=1, keepdim=True).values + 1e-10)
                    else: # random revoke
                        out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list
                        score = torch.ones_like(out_idx).float().cuda()

                    _, log_x_recon_tc = self.VQ_Diffusion.model.transformer.p_pred(log_z_tc, cond_emb, t)
                    out_tc = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon_tc) # recon -> sample -> x_0
                    out_idx_tc = log_onehot_to_index(out_tc) # b, 1024
                    out2_idx_tc = torch.full_like(out_idx_tc, self.num_classes-1).to(out_idx_tc.device) # all mask index list

                    if self.purity:
                        print(f"Purity & Token Critic step i = {i}")
                    else:
                        print(f"Random & Token Critic step i = {i}")

                    t_1 = torch.full((batch_size,), time_list[i+1], device=device, dtype=torch.long) # t-1 step
                    score_tc = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                    if tc_guidance != None:
                        cf_score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=tc_cf_cond_emb)
                        score_tc = cf_score + tc_guidance * (score_tc - cf_score)
                    score_tc = 1 - score_tc # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                    score_tc = score_tc.clamp(min = 0) # score clamp for CF minus probability
                    # score_tc = score_tc * score_weight
                    score_tc = (score_tc - score_tc.mean(1, keepdim=True)) * score_weight + score_tc.mean(1, keepdim=True)
                    score_tc = score_tc.clamp(min = 0)
                    score_tc += self.n_t(diffusion_index, a, b) # n(t) for randomness

                    if self.purity:
                        for ii in range(batch_size):
                            sel = torch.multinomial(score[ii], purity_mask[i])
                            # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                            out2_idx[ii][sel] = out_idx[ii][sel]
                        log_z = index_to_log_onehot(out2_idx, self.num_classes)
                    else:
                        for ii in range(batch_size):
                            sel = torch.multinomial(score[ii], n)
                            # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                            out2_idx[ii][sel] = out_idx[ii][sel]
                        log_z = index_to_log_onehot(out2_idx, self.num_classes)

                    out2_idx_tc = torch.full_like(out_idx_tc, self.num_classes-1).to(out_idx.device)
                    for ii in range(batch_size):
                        sel = torch.multinomial(score_tc[ii], n)
                        # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                        out2_idx_tc[ii][sel] = out_idx_tc[ii][sel]
                    log_z_tc = index_to_log_onehot(out2_idx_tc, self.num_classes)

                    if 'r' in log_set:
                        self.recon_image_log(out_idx)
                        self.recon_image_log_tc(out_idx_tc)

                    if 't' in log_set:
                        self.mask_token_log(out2_idx)
                        self.mask_token_log_tc(out2_idx_tc)
                    
            t = torch.full((batch_size,), time_list[-1], device=device, dtype=torch.long)

            _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
            out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
            content_token = log_onehot_to_index(out) # b, 1024

            _, log_x_recon_tc = self.VQ_Diffusion.model.transformer.p_pred(log_z_tc, cond_emb, t)
            out_tc = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon_tc) # recon -> sample -> x_0
            content_token_tc = log_onehot_to_index(out_tc) # b, 1024

            if 't' in log_set:
                self.mask_token_log(content_token)
                self.mask_token_log_tc(content_token_tc)
                
        
        # decoding token
        content = self.VQ_Diffusion.model.content_codec.decode(content_token)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        content_tc = self.VQ_Diffusion.model.content_codec.decode(content_token_tc)
        content_tc = content_tc.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result" : wandb.Image(im)})

        for b in range(content_tc.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content_tc[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result_tc" : wandb.Image(im)})

    @torch.no_grad()
    def draft_test(self, text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, noise_step=5, noise_power=5 ,wandb_log="s,t,r", a=0., b=0., score_weight=1., purity=True):
        """
        step: 16, 50, 100

        prepare condition: text -> tokenize -> CLIP -> cond_emb
        initialize: All mask
        
        iterate decoding step:
            1) VQ: 1 step reconstruction
            2) TC: Masking based on score

        purity
            True -> purity sampling
            False -> random revoke
        """
        log_set = wandb_log.split(',')
        save_root = 'RESULT'
        os.makedirs(save_root, exist_ok=True)
        str_cond = str(text) + '_' + str(step)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        # masking list & Diffusion time step
        self.purity = purity

        if step == 16:
            n_sample = [64] * 16
            time_list = [index for index in range(100 -5, -1, -6)]
        elif step == 50:
            n_sample = [10] + [21, 20] * 24 + [30]
            time_list = [index for index in range(100 -1, -1, -2)]
        else: # 100
            n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
            time_list = [index for index in range(100 -1, -1, -1)]

        purity_mask = n_sample
        n_mask = []
        for s in range(1, step):
            n_sample[s] += n_sample[s-1]
        for s in range(step):
            n_mask.append(1024 - n_sample[s]) # the number of masking tokens each step

        n_noise = [noise_power * 1024 // step] * noise_step
        t_noise = [time_list[-noise_power]] * noise_step


        # setting for VQ_Diffusion
        self.VQ_Diffusion.model.guidance_scale = vq_guidance
        self.VQ_Diffusion.model.learnable_cf = self.VQ_Diffusion.model.transformer.learnable_cf = True

        cf_cond_emb = self.VQ_Diffusion.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.tc_learnable: 
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1).float()
        else:
            none_text = [''] * batch_size
            condition_token = self.prepare_condition(none_text) # BPE token
            with torch.no_grad(): # condition(CLIP) -> freeze
                tc_cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
                tc_cf_cond_emb = tc_cond_emb.float() # CLIP condition

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
        self.VQ_Diffusion.model.truncation_forward = False

        # prepare condition: text -> tokenize -> CLIP -> cond_emb
        text_list = [text] * batch_size
        condition_token = self.prepare_condition(text_list) # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
            cond_emb = cond_emb.float() # CLIP condition

        # initialize: All mask 
        device = self.device
        zero_logits = torch.zeros((batch_size, self.num_classes-1, self.VQ_Diffusion.model.transformer.shape),device=device)
        one_logits = torch.ones((batch_size, 1, self.VQ_Diffusion.model.transformer.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        with torch.no_grad():
            # initialize with purity or random revoke
            for i, (n, diffusion_index) in tqdm(enumerate(zip(n_sample[:-1], time_list[:-1]))): # before last step
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024

                if self.purity:
                    log_z_idx = log_onehot_to_index(log_z)
                    out2_idx = log_z_idx.clone() # previous
                    score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                    score /= (score.max(dim=1, keepdim=True).values + 1e-10)
                else: # random revoke
                    out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list
                    score = torch.ones_like(out_idx).float().cuda()
                                
                if self.purity:
                    print(f"Purity initialize i = {i}")
                    for ii in range(batch_size):
                        sel = torch.multinomial(score[ii], purity_mask[i])
                        # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                        out2_idx[ii][sel] = out_idx[ii][sel]
                    log_z = index_to_log_onehot(out2_idx, self.num_classes)
                else:
                    print(f"Random initialize i = {i}")
                    for ii in range(batch_size):
                        sel = torch.multinomial(score[ii], n)
                        # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                        out2_idx[ii][sel] = out_idx[ii][sel]
                    log_z = index_to_log_onehot(out2_idx, self.num_classes)

            t = torch.full((batch_size,), time_list[-1], device=device, dtype=torch.long)
            _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
            out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
            out_idx = log_onehot_to_index(out) # b, 1024
            self.recon_image_log(out_idx)
            self.recon_image_log_tc(out_idx)

            out_idx_tc = out_idx.clone() # for token_critic

            for i, (num_noise, time_noise) in tqdm(enumerate(zip(n_noise, t_noise))):
                # random noise
                print(f"Revise step i = {i}")
                out_idx2 = out_idx.clone()
                score = torch.ones_like(out_idx).float().cuda()
                for ii in range(batch_size):
                    sel = torch.multinomial(score[ii], num_noise)
                    out_idx2[ii][sel] = self.num_classes - 1
                log_z = index_to_log_onehot(out_idx2, self.num_classes) # noise added

                t = torch.full((batch_size,), time_noise, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024 / recon

                # token critic
                score_tc = self.Token_Critic.inference_score(input={'t': t, 'recon_token': out_idx_tc}, cond_emb=cond_emb) # b, 1024
                if tc_guidance != None:
                    cf_score = self.Token_Critic.inference_score(input={'t': t, 'recon_token': out_idx_tc}, cond_emb=tc_cf_cond_emb)
                    score_tc = cf_score + tc_guidance * (score_tc - cf_score)
                # score_tc = 1 - score_tc # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                score_tc = score_tc.clamp(min = 0) # score clamp for CF minus probability => not confident score
                # score_tc = score_tc * score_weight
                score_tc = (score_tc - score_tc.mean(1, keepdim=True)) * score_weight + score_tc.mean(1, keepdim=True)
                score_tc = score_tc.clamp(min = 0)
                score_tc += self.n_t(diffusion_index, a, b) # n(t) for randomness
                out_idx2_tc = out_idx_tc.clone()
                for ii in range(batch_size):
                    sel = torch.multinomial(score_tc[ii], num_noise)
                    out_idx2_tc[ii][sel] = self.num_classes - 1
                log_z_tc = index_to_log_onehot(out_idx2_tc, self.num_classes) # noise added

                _, log_x_recon_tc = self.VQ_Diffusion.model.transformer.p_pred(log_z_tc, cond_emb, t)
                out_tc = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon_tc) # recon -> sample -> x_0
                out_idx_tc = log_onehot_to_index(out_tc) # b, 1024 / recon

                if 'r' in log_set:
                    self.recon_image_log(out_idx)
                    self.recon_image_log_tc(out_idx_tc)

                if 't' in log_set:
                    self.mask_token_log(out_idx2)
                    self.mask_token_log_tc(out_idx2_tc)

        # decoding token
        content = self.VQ_Diffusion.model.content_codec.decode(out_idx)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        content_tc = self.VQ_Diffusion.model.content_codec.decode(out_idx_tc)
        content_tc = content_tc.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result" : wandb.Image(im)})

        for b in range(content_tc.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content_tc[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result_tc" : wandb.Image(im)})


    @torch.no_grad()
    def draft_test(self, text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, noise_step=5, noise_power=5 ,wandb_log="s,t,r", a=0., b=0., score_weight=1., purity=True):
        """
        step: 16, 50, 100

        prepare condition: text -> tokenize -> CLIP -> cond_emb
        initialize: All mask
        
        iterate decoding step:
            1) VQ: 1 step reconstruction
            2) TC: Masking based on score

        purity
            True -> purity sampling
            False -> random revoke
        """
        log_set = wandb_log.split(',')
        save_root = 'RESULT'
        os.makedirs(save_root, exist_ok=True)
        str_cond = str(text) + '_' + str(step)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        # masking list & Diffusion time step
        self.purity = purity

        if step == 16:
            n_sample = [64] * 16
            time_list = [index for index in range(100 -5, -1, -6)]
        elif step == 50:
            n_sample = [10] + [21, 20] * 24 + [30]
            time_list = [index for index in range(100 -1, -1, -2)]
        else: # 100
            n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
            time_list = [index for index in range(100 -1, -1, -1)]

        purity_mask = n_sample
        n_mask = []
        for s in range(1, step):
            n_sample[s] += n_sample[s-1]
        for s in range(step):
            n_mask.append(1024 - n_sample[s]) # the number of masking tokens each step

        n_noise = [noise_power * 1024 // step] * noise_step
        t_noise = [time_list[-noise_power]] * noise_step


        # setting for VQ_Diffusion
        self.VQ_Diffusion.model.guidance_scale = vq_guidance
        self.VQ_Diffusion.model.learnable_cf = self.VQ_Diffusion.model.transformer.learnable_cf = True

        cf_cond_emb = self.VQ_Diffusion.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.tc_learnable: 
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1).float()
        else:
            none_text = [''] * batch_size
            condition_token = self.prepare_condition(none_text) # BPE token
            with torch.no_grad(): # condition(CLIP) -> freeze
                tc_cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
                tc_cf_cond_emb = tc_cond_emb.float() # CLIP condition

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
        self.VQ_Diffusion.model.truncation_forward = False

        # prepare condition: text -> tokenize -> CLIP -> cond_emb
        text_list = [text] * batch_size
        condition_token = self.prepare_condition(text_list) # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
            cond_emb = cond_emb.float() # CLIP condition

        # initialize: All mask 
        device = self.device
        zero_logits = torch.zeros((batch_size, self.num_classes-1, self.VQ_Diffusion.model.transformer.shape),device=device)
        one_logits = torch.ones((batch_size, 1, self.VQ_Diffusion.model.transformer.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        with torch.no_grad():
            # initialize with purity or random revoke
            for i, (n, diffusion_index) in tqdm(enumerate(zip(n_sample[:-1], time_list[:-1]))): # before last step
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024

                if self.purity:
                    log_z_idx = log_onehot_to_index(log_z)
                    out2_idx = log_z_idx.clone() # previous
                    score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                    score /= (score.max(dim=1, keepdim=True).values + 1e-10)
                else: # random revoke
                    out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list
                    score = torch.ones_like(out_idx).float().cuda()
                                
                if self.purity:
                    print(f"Purity initialize i = {i}")
                    for ii in range(batch_size):
                        sel = torch.multinomial(score[ii], purity_mask[i])
                        # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                        out2_idx[ii][sel] = out_idx[ii][sel]
                    log_z = index_to_log_onehot(out2_idx, self.num_classes)
                else:
                    print(f"Random initialize i = {i}")
                    for ii in range(batch_size):
                        sel = torch.multinomial(score[ii], n)
                        # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                        out2_idx[ii][sel] = out_idx[ii][sel]
                    log_z = index_to_log_onehot(out2_idx, self.num_classes)

            t = torch.full((batch_size,), time_list[-1], device=device, dtype=torch.long)
            _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
            out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
            out_idx = log_onehot_to_index(out) # b, 1024
            self.recon_image_log(out_idx)
            self.recon_image_log_tc(out_idx)

            out_idx_tc = out_idx.clone() # for token_critic

            for i, (num_noise, time_noise) in tqdm(enumerate(zip(n_noise, t_noise))):
                # random noise
                print(f"Revise step i = {i}")
                out_idx2 = out_idx.clone()
                score = torch.ones_like(out_idx).float().cuda()
                for ii in range(batch_size):
                    sel = torch.multinomial(score[ii], num_noise)
                    out_idx2[ii][sel] = self.num_classes - 1
                log_z = index_to_log_onehot(out_idx2, self.num_classes) # noise added

                t = torch.full((batch_size,), time_noise, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024 / recon

                # token critic
                score_tc = self.Token_Critic.inference_score(input={'t': t, 'recon_token': out_idx_tc}, cond_emb=cond_emb) # b, 1024
                if tc_guidance != None:
                    cf_score = self.Token_Critic.inference_score(input={'t': t, 'recon_token': out_idx_tc}, cond_emb=tc_cf_cond_emb)
                    score_tc = cf_score + tc_guidance * (score_tc - cf_score)
                # score_tc = 1 - score_tc # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                score_tc = score_tc.clamp(min = 0) # score clamp for CF minus probability => not confident score
                # score_tc = score_tc * score_weight
                score_tc = (score_tc - score_tc.mean(1, keepdim=True)) * score_weight + score_tc.mean(1, keepdim=True)
                score_tc = score_tc.clamp(min = 0)
                score_tc += self.n_t(diffusion_index, a, b) # n(t) for randomness
                out_idx2_tc = out_idx_tc.clone()
                for ii in range(batch_size):
                    sel = torch.multinomial(score_tc[ii], num_noise)
                    out_idx2_tc[ii][sel] = self.num_classes - 1
                log_z_tc = index_to_log_onehot(out_idx2_tc, self.num_classes) # noise added

                _, log_x_recon_tc = self.VQ_Diffusion.model.transformer.p_pred(log_z_tc, cond_emb, t)
                out_tc = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon_tc) # recon -> sample -> x_0
                out_idx_tc = log_onehot_to_index(out_tc) # b, 1024 / recon

                if 'r' in log_set:
                    self.recon_image_log(out_idx)
                    self.recon_image_log_tc(out_idx_tc)

                if 't' in log_set:
                    self.mask_token_log(out_idx2)
                    self.mask_token_log_tc(out_idx2_tc)

        # decoding token
        content = self.VQ_Diffusion.model.content_codec.decode(out_idx)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        content_tc = self.VQ_Diffusion.model.content_codec.decode(out_idx_tc)
        content_tc = content_tc.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result" : wandb.Image(im)})

        for b in range(content_tc.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content_tc[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result_tc" : wandb.Image(im)})

    @torch.no_grad()
    def inpaint_test(self, text, text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, noise_step=5, noise_power=5 ,wandb_log="s,t,r", a=0., b=0., score_weight=1., purity=True):
        """
        step: 16, 50, 100

        prepare condition: text -> tokenize -> CLIP -> cond_emb
        initialize: All mask
        
        iterate decoding step:
            1) VQ: 1 step reconstruction
            2) TC: Masking based on score

        purity
            True -> purity sampling
            False -> random revoke
        """
        log_set = wandb_log.split(',')
        save_root = 'RESULT'
        os.makedirs(save_root, exist_ok=True)
        str_cond = str(text) + '_' + str(step)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        # masking list & Diffusion time step
        self.purity = purity

        if step == 16:
            n_sample = [64] * 16
            time_list = [index for index in range(100 -5, -1, -6)]
        elif step == 50:
            n_sample = [10] + [21, 20] * 24 + [30]
            time_list = [index for index in range(100 -1, -1, -2)]
        else: # 100
            n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
            time_list = [index for index in range(100 -1, -1, -1)]

        purity_mask = n_sample
        n_mask = []
        for s in range(1, step):
            n_sample[s] += n_sample[s-1]
        for s in range(step):
            n_mask.append(1024 - n_sample[s]) # the number of masking tokens each step

        n_noise = [noise_power * 1024 // step] * noise_step
        t_noise = [time_list[-noise_power]] * noise_step


        # setting for VQ_Diffusion
        self.VQ_Diffusion.model.guidance_scale = vq_guidance
        self.VQ_Diffusion.model.learnable_cf = self.VQ_Diffusion.model.transformer.learnable_cf = True

        cf_cond_emb = self.VQ_Diffusion.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.tc_learnable: 
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1).float()
        else:
            none_text = [''] * batch_size
            condition_token = self.prepare_condition(none_text) # BPE token
            with torch.no_grad(): # condition(CLIP) -> freeze
                tc_cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
                tc_cf_cond_emb = tc_cond_emb.float() # CLIP condition

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
        self.VQ_Diffusion.model.truncation_forward = False

        # prepare condition: text -> tokenize -> CLIP -> cond_emb
        text_list = [text] * batch_size
        text_list2 = [text2] * batch_size
        condition_token = self.prepare_condition(text_list) # BPE token
        condition_token2 = self.prepare_condition(text_list2) # BPE token
        with torch.no_grad(): # condition(CLIP) -> freeze
            cond_emb = self.VQ_Diffusion.model.transformer.condition_emb(condition_token['condition_token']) # B x Ld x D   256*1024
            cond_emb = cond_emb.float() # CLIP condition
            cond_emb2 = self.VQ_Diffusion.model.transformer.condition_emb(condition_token2['condition_token']) # B x Ld x D   256*1024
            cond_emb2 = cond_emb2.float() # CLIP condition

        # initialize: All mask 
        device = self.device
        zero_logits = torch.zeros((batch_size, self.num_classes-1, self.VQ_Diffusion.model.transformer.shape),device=device)
        one_logits = torch.ones((batch_size, 1, self.VQ_Diffusion.model.transformer.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        with torch.no_grad():
            # initialize with purity or random revoke
            for i, (n, diffusion_index) in tqdm(enumerate(zip(n_sample[:-1], time_list[:-1]))): # before last step
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024

                # if self.purity:
                #     log_z_idx = log_onehot_to_index(log_z)
                #     out2_idx = log_z_idx.clone() # previous
                #     score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                #     score /= (score.max(dim=1, keepdim=True).values + 1e-10)
                # else: # random revoke
                #     out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list
                #     score = torch.ones_like(out_idx).float().cuda()
                                
                # if self.purity:
                #     print(f"Purity initialize i = {i}")
                #     for ii in range(batch_size):
                #         sel = torch.multinomial(score[ii], purity_mask[i])
                #         # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                #         out2_idx[ii][sel] = out_idx[ii][sel]
                #     log_z = index_to_log_onehot(out2_idx, self.num_classes)
                # else:
                #     print(f"Random initialize i = {i}")
                #     for ii in range(batch_size):
                #         sel = torch.multinomial(score[ii], n)
                #         # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                #         out2_idx[ii][sel] = out_idx[ii][sel]
                #     log_z = index_to_log_onehot(out2_idx, self.num_classes)
                t_1 = torch.full((batch_size,), time_list[i+1], device=device, dtype=torch.long) # t-1 step
                out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list
                score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                if tc_guidance != None:
                    cf_score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=tc_cf_cond_emb)
                    score = cf_score + 2 * (score - cf_score)
                score = 1 - score # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                score = score.clamp(min = 0) # score clamp for CF minus probability
                # score = score * score_weight
                score = (score - score.mean(1, keepdim=True)) * score_weight + score.mean(1, keepdim=True)
                score = score.clamp(min = 0)
                score += self.n_t(diffusion_index, a, b) # n(t) for randomness

                for ii in range(batch_size):
                    sel = torch.multinomial(score[ii], n)
                    # _, sel = torch.topk(score[ii], n, dim=0) # determinisitic
                    out2_idx[ii][sel] = out_idx[ii][sel]
                log_z = index_to_log_onehot(out2_idx, self.num_classes)


            t = torch.full((batch_size,), time_list[-1], device=device, dtype=torch.long)
            _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
            out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
            out_idx = log_onehot_to_index(out) # b, 1024
            self.recon_image_log(out_idx)
            self.recon_image_log_tc(out_idx)

            out_idx_tc = out_idx.clone() # for token_critic

            for i, (num_noise, time_noise) in tqdm(enumerate(zip(n_noise, t_noise))):
                # random noise
                print(f"Revise step i = {i}")
                out_idx2 = out_idx.clone()
                score = torch.ones_like(out_idx).float().cuda()
                for ii in range(batch_size):
                    sel = torch.multinomial(score[ii], num_noise)
                    out_idx2[ii][sel] = self.num_classes - 1
                log_z = index_to_log_onehot(out_idx2, self.num_classes) # noise added

                t = torch.full((batch_size,), time_noise, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb2, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024 / recon

                # token critic
                score_tc = self.Token_Critic.inference_score(input={'t': t, 'recon_token': out_idx_tc}, cond_emb=cond_emb2) # b, 1024
                if tc_guidance != None:
                    cf_score = self.Token_Critic.inference_score(input={'t': t, 'recon_token': out_idx_tc}, cond_emb=tc_cf_cond_emb)
                    score_tc = cf_score + tc_guidance * (score_tc - cf_score)
                # score_tc = 1 - score_tc # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)
                score_tc = score_tc.clamp(min = 0) # score clamp for CF minus probability => not confident score
                # score_tc = score_tc * score_weight
                score_tc = (score_tc - score_tc.mean(1, keepdim=True)) * score_weight + score_tc.mean(1, keepdim=True)
                score_tc = score_tc.clamp(min = 0)
                score_tc += self.n_t(diffusion_index, a, b) # n(t) for randomness
                out_idx2_tc = out_idx_tc.clone()
                for ii in range(batch_size):
                    sel = torch.multinomial(score_tc[ii], num_noise)
                    out_idx2_tc[ii][sel] = self.num_classes - 1
                log_z_tc = index_to_log_onehot(out_idx2_tc, self.num_classes) # noise added

                _, log_x_recon_tc = self.VQ_Diffusion.model.transformer.p_pred(log_z_tc, cond_emb2, t)
                out_tc = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon_tc) # recon -> sample -> x_0
                out_idx_tc = log_onehot_to_index(out_tc) # b, 1024 / recon

                if 'r' in log_set:
                    self.recon_image_log(out_idx)
                    self.recon_image_log_tc(out_idx_tc)

                if 't' in log_set:
                    self.mask_token_log(out_idx2)
                    self.mask_token_log_tc(out_idx2_tc)

                if 's' in log_set:
                    # self.score_matrix_log(score)
                    self.score_matrix_log_tc(score_tc)
                    self.recon_image_with_attn_log(out_idx_tc, 1 - score_tc)

        # decoding token
        content = self.VQ_Diffusion.model.content_codec.decode(out_idx)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        content_tc = self.VQ_Diffusion.model.content_codec.decode(out_idx_tc)
        content_tc = content_tc.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result" : wandb.Image(im)})

        for b in range(content_tc.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content_tc[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
            wandb.log({"result_tc" : wandb.Image(im)})


if __name__ == '__main__':
    model = VQ_Critic(vq='coco', tc_config='configs/token_critic.yaml', tc_learnable_cf=False)
    model.load_tc('/content/drive/MyDrive/VQ/Improved_VQ-Diffusion/tc_ckpt/685_10.pth')
    # wandb.init(project='TC test', name = '0_scracth_12_teddybear_16')
    # model.inference_generate_sample_with_condition(text="teddy bear playing in the pool", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='s,r,t')
    # wandb.finish()
    # wandb.init(project='accuracy_test', name = '427_1')
    # model.accuracy_test()

    # text = 'A young boy holding a baseball bat while standing in front of a building.'
    # text = 'A teddy bear and a very old sewing machine are shown.'
    # text = "A cat laying on a computer desk next to a laptop."
    # text = "A black bear walking down a rocky mountain side."
    # text = 'A black microwave oven with toys sitting on top of it.'
    # text = 'A slice of pizza on a plate sitting on a table'
    # text = 'A slice of pizza on a plate sitting on a table'
    # 'A picture of a teddy bear on a stone.'

    # text = 'A picture of a teddy bear on a stone.'

    # wandb.init(project='685_10', name = '16_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='685_10', name = '16_00_3')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3, step=16, wandb_log='',a=0, b=0)
    # wandb.finish()
    
    # texts = ['a bicycle replica with a clock as the front wheel.', 'a black honda motorcycle parked in front of a garage.', 
    # 'a room with blue walls and a white sink and door.', 'a car that seems to be parked illegally behind a legally parked car', 'a large passenger airplane flying through the air.', 'there is a gol plane taking off in a partly cloudy sky.',
    # 'blue and white color scheme in a small bathroom.', 'this is a blue and white bathroom with a wall sink and a lifesaver on the wall.', 'a blue boat themed bathroom with a life preserver on the wall', 'the bike has a clock as a tire.',
    # 'a honda motorcycle parked in a grass driveway', 'two cars parked on the sidewalk on the street', 'an airplane that is, either, landing or just taking off.']
    # texts = ['A teddy bear and a very old sewing machine are shown.', 'A slice of pizza on a plate sitting on a table', 'A teddy bear playing in the pool',
    # "A black bear walking down a rocky mountain side.", "a close up of a person jumping skis in the air", "A group of people competing in a cross country skiing race.", "The colorful roses are in a clear glass jar."]
    texts = ['A baseball player in red shorts prepares to swing at the ball.', 'A cellular phone next to a purse sitting on a table.', 'The two children have a large teddy bear.',
    'grapes broccoli and berries in a plastic container', 'Two people wearing helmets skate downhill on skate boards', 'A sectioned lunch box with a sandwich, vegetables, apple and dessert']
    for text in texts:

    # text = 'a car that seems to be parked illegally behind a legally parked car'
        wandb.init(project='685_10', name = text + '_685_10_det')
        wandb.finish()

        wandb.init(project='685_10', name = '16_00_0_a0')
        model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 0)
        wandb.finish()

        # wandb.init(project='685_10', name = '16_00_3_a0')
        # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 0)
        # wandb.finish()

        wandb.init(project='685_10', name = '16_00_0_a1')
        model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 1.)
        wandb.finish()

        # wandb.init(project='685_10', name = '16_00_3_a1')
        # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 1.)
        # wandb.finish()

        wandb.init(project='685_10', name = '16_00_0_a2')
        model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 2.)
        wandb.finish()

        wandb.init(project='685_10', name = '16_00_3_a2')
        model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 2.)
        wandb.finish()

        wandb.init(project='685_10', name = '16_00_0_a3')
        model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 3.)
        wandb.finish()

        wandb.init(project='685_10', name = '16_00_3_a3')
        model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 3.)
        wandb.finish()

        wandb.init(project='685_10', name = '16_00_3_a4')
        model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3, step=16, wandb_log='',a=0, b=0, gaussian_alpha = 4.)
        wandb.finish()

    # wandb.init(project='685_10', name = '16_105_5')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5, step=16, wandb_log='',a=0.1, b=0.05)
    # wandb.finish()

    # wandb.init(project='685_10', name = '16_00_9')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9, step=16, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='685_10', name = '50_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='685_10', name = '100_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=100, wandb_log='',a=0, b=0)
    # wandb.finish()




    # wandb.init(project='685_10', name = 'refine_16step_5start_weight1_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, start_step = 5, wandb_log='r', a=0, b=0, score_weight=1., purity=True)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'refine_16step_8start_weight1_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, start_step = 8, wandb_log='r', a=0, b=0, score_weight=1., purity=True)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'refine_16step_11start_weight1_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, start_step = 11, wandb_log='r', a=0, b=0, score_weight=1., purity=True)
    # wandb.finish()


    # wandb.init(project='685_10', name = 'refine_50step_15start_weight1_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, start_step = 15, wandb_log='r', a=0, b=0, score_weight=1., purity=True)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'refine_50step_25start_weight1_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, start_step = 25, wandb_log='r', a=0, b=0, score_weight=1., purity=True)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'refine_50step_35start_weight1_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, start_step = 35, wandb_log='r', a=0, b=0, score_weight=1., purity=True)
    # wandb.finish()

    # # # # 16
    # wandb.init(project='685_10', name = 'draft_16step_10_5_weight1_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, noise_step=10, noise_power=5, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'draft_16step_10_7_weight1_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, noise_step=10, noise_power=7, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'draft_16step_10_9_weight1_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, noise_step=10, noise_power=9, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # # # # # 50

    # wandb.init(project='685_10', name = 'draft_50step_10_15_weight1_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=10, noise_power=15, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'draft_50step_10_20_weight1_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=10, noise_power=20, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'draft_50step_10_30_weight1_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=10, noise_power=30, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # # ## weight

    # # wandb.init(project='TC draft2', name = 'draft_50step_5_20_weight2_g0_r')
    # # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=2, noise_power=25, wandb_log='r', a=0, b=0, score_weight=2., purity=False)
    # # wandb.finish()

    # # wandb.init(project='TC draft2', name = 'draft_50step_5_20_weight3_g0_r')
    # # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=2, noise_power=25, wandb_log='r', a=0, b=0, score_weight=3., purity=False)
    # # wandb.finish()

    # # wandb.init(project='TC draft2', name = 'draft_50step_5_10_weight4_g0_r')
    # # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=2, noise_power=25, wandb_log='r', a=0, b=0, score_weight=4., purity=False)
    # # wandb.finish()


    # # # ## guidance

    # wandb.init(project='685_10', name = 'draft_50step_2_25_weight1_g0_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=2, noise_power=25, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'draft_50step_2_25_weight1_g3_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, noise_step=2, noise_power=25, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'draft_50step_2_25_weight1_g5_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=50, noise_step=2, noise_power=25, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'draft_50step_2_25_weight1_g9_r')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9., step=50, noise_step=2, noise_power=25, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()


    # text = 'A black bear walking down a rocky mountain side.'
    # text2 = 'A white bear walking down a rocky mountain side.'

    # # text = 'The blue and yellow train is traveling down the tracks.'
    # # text2 = 'The blue and red train is traveling down the tracks.'

    # text = 'a close up of a single zebra standing on a dirt area.'
    # text2 = 'a close up of a single dog standing on a dirt area.'

    # wandb.init(project='685_10', name = text + ' ' + text2)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'paint_50step_15_40_weight1_g0_r')
    # model.inpaint_test(text=text, text2 = text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=15, noise_power=40, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'paint_50step_15_25_weight3_g0_r')
    # model.inpaint_test(text=text, text2 = text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=15, noise_power=25, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'paint_50step_15_25_weight4_g0_r')
    # model.inpaint_test(text=text, text2 = text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=15, noise_power=25, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()


    # wandb.init(project='685_10', name = 'paint_50step_15_40_weight1_g0_r')
    # model.inpaint_test(text=text, text2 = text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, noise_step=15, noise_power=40, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'paint_50step_15_40_weight1_g3_r')
    # model.inpaint_test(text=text, text2 = text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, noise_step=15, noise_power=40, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'paint_50step_15_40_weight1_g5_r')
    # model.inpaint_test(text=text, text2 = text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=50, noise_step=15, noise_power=40, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()

    # wandb.init(project='685_10', name = 'paint_50step_15_40_weight1_g2_r')
    # model.inpaint_test(text=text, text2 = text2, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9., step=50, noise_step=15, noise_power=40, wandb_log='r', a=0, b=0, score_weight=1., purity=False)
    # wandb.finish()










    # wandb.init(project='TC draft2', name = 'draft_100step_10_30_weight1_g0_purity')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=100, noise_step=10, noise_power=30, wandb_log='r', a=0, b=0, score_weight=3., purity=True)
    # wandb.finish()

    # wandb.init(project='TC draft2', name = 'draft_100step_10_30_weight1_g3_purity')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=100, noise_step=10, noise_power=30, wandb_log='r', a=0, b=0, score_weight=3., purity=True)
    # wandb.finish()

    # wandb.init(project='TC draft2', name = 'draft_100step_10_30_weight1_g5_purity')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=100, noise_step=10, noise_power=30, wandb_log='r', a=0, b=0, score_weight=3., purity=True)
    # wandb.finish()

    # wandb.init(project='TC draft2', name = 'draft_100step_10_30_weight1_g9_purity')
    # model.draft_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9., step=100, noise_step=10, noise_power=30, wandb_log='r', a=0, b=0, score_weight=3., purity=True)
    # wandb.finish()

    # wandb.init(project='TC refine', name = 'refine_16step_8start_weight2_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, start_step = 8, wandb_log='r', a=0, b=0, score_weight=2., purity=True)
    # wandb.finish()

    # wandb.init(project='TC refine', name = 'refine_16step_11start_weight2_purity')
    # model.refine_test(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, start_step = 11, wandb_log='r', a=0, b=0, score_weight=2., purity=True)
    # wandb.finish()


    # wandb.init(project='TC test 1024', name = 'rand_16')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=16, wandb_log='',a=0, b=100, token_critic=False)
    # wandb.finish()

    # wandb.init(project='TC test 1024', name = 'rand_50')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=50, wandb_log='',a=0, b=100, token_critic=False)
    # wandb.finish()

    # wandb.init(project='TC test 1024', name = 'rand_100')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=100, wandb_log='',a=0, b=100, token_critic=False)
    # wandb.finish()


    # wandb.init(project='TC test 1024', name = '-----427_13-----')
    # wandb.finish()
    
    # wandb.init(project='TC test 1024', name = '16_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test 1024', name = '50_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # # wandb.init(project='TC test 1024', name = '100_00')
    # # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=100, wandb_log='',a=0, b=0)
    # # wandb.finish()

    # wandb.init(project='TC test 1024', name = '16_cf5.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=16, wandb_log='',a=0.0, b=0.0)
    # wandb.finish()

    # wandb.init(project='TC test 1024', name = '16_cf9.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9., step=16, wandb_log='',a=0.0, b=0.0)
    # wandb.finish()

    # wandb.init(project='TC test 1024', name = '50_cf5.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=50, wandb_log='',a=0.0, b=0.0)
    # wandb.finish()

    # wandb.init(project='TC test 1024', name = '50_cf9.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9., step=50, wandb_log='',a=0.0, b=0.0)
    # wandb.finish()

    # wandb.init(project='TC test 1021-2', name = 'train_16_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='r',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test front & back', name = '16_rotation_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='r',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test front & back', name = '50_rotation_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='r',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test front & back', name = '100_rotation_00')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=100, wandb_log='r',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test front & back', name = 'train_50_front')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='r',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1021-2', name = 'train_16_cf5.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=16, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1021-2', name = 'train_16_cf9.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9., step=16, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1021-2', name = 'train_50_cf5.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=50, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1021-2', name = 'train_50_cf9.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=9., step=50, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1019', name = 'train_16_10')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0.1, b=0)
    # wandb.finish()

    # wandb.init(project='TC test 1019', name = 'train_16_01')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='',a=0, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1019', name = 'train_100')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=100, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    

    
    # 16 step guidacne test

    # wandb.init(project='TC test 1019', name = 'train_16_cf3.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=16, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1019', name = 'train_16_cf5.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=16, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test 1019', name = 'train_16_cf7.0')
    # model.inference_generate_sample_with_condition(text=text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=7., step=16, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = 'train_50')
    # model.inference_generate_sample_with_condition(text='teddy bear playing in the pool', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = 'train_50_cf3.0')
    # model.inference_generate_sample_with_condition(text='teddy bear playing in the pool', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = 'train_50_cf5.0')
    # model.inference_generate_sample_with_condition(text='teddy bear playing in the pool', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = 'train_50_cf7.0')
    # model.inference_generate_sample_with_condition(text='teddy bear playing in the pool', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=7., step=50, wandb_log='',a=0, b=0)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = 'train_100')
    # # model.inference_generate_sample_with_condition(text='A woman and child are boarding a colorful little train.', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=100, wandb_log='',a=0.1, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = 'train_100_cf3.0')
    # # model.inference_generate_sample_with_condition(text='A woman and child are boarding a colorful little train.', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=100, wandb_log='',a=0.1, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = 'train_100_cf5.0')
    # # model.inference_generate_sample_with_condition(text='A woman and child are boarding a colorful little train.', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=5., step=100, wandb_log='',a=0.1, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = 'train_100_cf7.0')
    # # model.inference_generate_sample_with_condition(text='A woman and child are boarding a colorful little train.', batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=7., step=100, wandb_log='',a=0.1, b=0.1)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '---A woman and child are boarding a colorful little train.---')
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_rand(b100)')
    # model.inference_generate_sample_with_condition(text="A woman and child are boarding a colorful little train.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=100)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00')
    # model.inference_generate_sample_with_condition(text="A woman and child are boarding a colorful little train.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00_cf3')
    # model.inference_generate_sample_with_condition(text="A woman and child are boarding a colorful little train.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00_cf7')
    # model.inference_generate_sample_with_condition(text="A woman and child are boarding a colorful little train.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=7., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()


    # wandb.init(project='TC test_2', name = '---A black and white picture of two men in suits and white tophats.---')
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_rand(b100)')
    # model.inference_generate_sample_with_condition(text="A black and white picture of two men in suits and white tophats.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=100)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00')
    # model.inference_generate_sample_with_condition(text="A black and white picture of two men in suits and white tophats.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00_cf3')
    # model.inference_generate_sample_with_condition(text="A black and white picture of two men in suits and white tophats.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00_cf7')
    # model.inference_generate_sample_with_condition(text="A black and white picture of two men in suits and white tophats.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=7., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '---A white bird with a long black peak standing near the ocean.---')
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_rand(b100)')
    # model.inference_generate_sample_with_condition(text="A white bird with a long black peak standing near the ocean.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=100)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00')
    # model.inference_generate_sample_with_condition(text="A white bird with a long black peak standing near the ocean.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00_cf3')
    # model.inference_generate_sample_with_condition(text="A white bird with a long black peak standing near the ocean.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()

    # wandb.init(project='TC test_2', name = '1ep_50_ab_00_cf7')
    # model.inference_generate_sample_with_condition(text="A white bird with a long black peak standing near the ocean.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=7., step=50, wandb_log='',a=0, b=0)
    # wandb.finish()




    # # wandb.init(project='TC test_2', name = '1ep_50_ab_11_cf3')
    # # model.inference_generate_sample_with_condition(text="a black metal bicycle with a clock inside the front wheel.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0.1, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = '----------------')
    # # wandb.finish()



    # # wandb.init(project='TC test_2', name = '1ep_50_ab_11_cf7')
    # # model.inference_generate_sample_with_condition(text="a black metal bicycle with a clock inside the front wheel.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=7., step=50, wandb_log='',a=0.1, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = '1ep_teddybear_50_ab_12_cf3')
    # # model.inference_generate_sample_with_condition(text="teddy bear playing in the pool", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0.1, b=0.2)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = '1ep_teddybear_50_ab_13_cf3')
    # # model.inference_generate_sample_with_condition(text="teddy bear playing in the pool", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0.1, b=0.3)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = '1ep_teddybear_50_ab_21_cf3')
    # # model.inference_generate_sample_with_condition(text="teddy bear playing in the pool", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0.2, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test_2', name = '1ep_teddybear_50_ab_31_cf3')
    # # model.inference_generate_sample_with_condition(text="teddy bear playing in the pool", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=50, wandb_log='',a=0.3, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test', name = '0_scratch_12_cat_ab_11_g3')
    # # model.inference_generate_sample_with_condition(text="a black cat is inside a white toilet.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0.1, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test', name = '0_scratch_12_cat_50_ab_21_g3')
    # # model.inference_generate_sample_with_condition(text="a black cat is inside a white toilet.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0.2, b=0.1)
    # # wandb.finish()

    # # wandb.init(project='TC test', name = '0_scratch_12_cat_50_ab_31_g3')
    # # model.inference_generate_sample_with_condition(text="a black cat is inside a white toilet.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0.3, b=0.1)
    # # wandb.finish()
    
    # # wandb.init(project='TC test', name = '0_scratch_12_cat_50_ab_12_g3')
    # # model.inference_generate_sample_with_condition(text="a black cat is inside a white toilet.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0.1, b=0.2)
    # # wandb.finish()

    # # wandb.init(project='TC test', name = '0_scratch_12_cat_50_ab_13_g3')
    # # model.inference_generate_sample_with_condition(text="a black cat is inside a white toilet.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=3., step=50, wandb_log='',a=0.1, b=0.3)
    # # wandb.finish()
