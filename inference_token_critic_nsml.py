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

import scipy.linalg
from tqdm.auto import tqdm

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.utils.misc import get_model_parameters_info
import image_synthesis.modeling.modules.clip.clip as clip
from image_synthesis.data.mscoco_dataset import CocoDataset 
from image_synthesis.engine.lr_scheduler import ReduceLROnPlateauWithWarmup

import nsml
from nsml import IS_ON_NSML
from nsml_utils import bind_model
clip_model_path = os.path.join(nsml.DATASET_PATH[1], 'train/ViT-B-32.pt')
diffusion_model_path = os.path.join(nsml.DATASET_PATH[2], 'train')
diffusion_model_name = os.listdir(diffusion_model_path)[0]
diffusion_model_path = os.path.join(diffusion_model_path, diffusion_model_name)
vqvae_model_path = os.path.join(nsml.DATASET_PATH[3], 'train')
vqvae_model_name = os.listdir(vqvae_model_path)[0]
vqvae_model_path = os.path.join(vqvae_model_path, vqvae_model_name)

import wandb
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

class VQ_Diffusion():
    def __init__(self, config, path=None, imagenet_cf=False):
        self.model = self.get_model(config_path=config, imagenet_cf=imagenet_cf)
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad=False

    def get_model(self, config_path, imagenet_cf):
        model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)

        if imagenet_cf:
            config['model']['params']['diffusion_config']['params']['transformer_config']['params']['class_number'] = 1001

        model = build_model(config)

        state_dict = torch.load(diffusion_model_path, map_location='cpu')

        missing, unexpected = model.load_state_dict(state_dict['model'], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        return model

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size, infer_speed=False, guidance_scale=1.0):
        self.model.guidance_scale = guidance_scale

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        if not IS_ON_NSML:
            os.makedirs(save_root, exist_ok=True)
            str_cond = str(condition)
            save_root_ = os.path.join(save_root, str_cond)
            os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+'r',
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im.save(save_path)
            

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, text2=False, purity_temp=1., ):
        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = learnable_cf # whether to use learnable classifier-free
        self.model.transformer.prior_rule = prior_rule      # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.model.transformer.prior_weight = prior_weight  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion
        self.model.transformer.purity_temp = purity_temp
        self.model.transformer.mask_schedule_test = 0
        data_i = {}
        data_i['text'] = [text]

        composition = False
        if text2 != False:
            data_i['text2'] = [text2]
            composition = True
        data_i['image'] = None
        condition = text

        if not IS_ON_NSML:
            os.makedirs(save_root, exist_ok=True)
            str_cond = str(condition)
            save_root_ = os.path.join(save_root, str_cond)
            os.makedirs(save_root_, exist_ok=True)

        if infer_speed != False:
            add_string = 'r,time'+str(infer_speed)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
                composition=composition # text2 condition yes or no
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)
        
        # recon log step by step
        # for i in range(9):
        #     content = model_out[f"{i}_step_token"]
        #     content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        #     for b in range(content.shape[0]):
        #         im = Image.fromarray(content[b])
        #         wandb.log({f"{i}_step recon" : wandb.Image(im)})


    def inference_generate_sample_for_clip_score(self, truncation_rate, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, text2=False, purity_temp=1., schedule=0):
        """
        T = 16 fix
        schedule = 1 ~ 4
        1) out -> in
        2) in -> out
        3) grid: blockwise
        4) grid: uniform

        5) random (fast)
        """

        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = learnable_cf # whether to use learnable classifier-free
        self.model.transformer.prior_rule = prior_rule      # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.model.transformer.prior_weight = prior_weight  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion
        self.model.transformer.purity_temp = purity_temp
        self.model.transformer.mask_schedule_test = schedule

        if infer_speed != False:
            add_string = 'r,time'+str(infer_speed)
        else:
            add_string = 'r'

        self.model.eval()
        cf_cond_emb = self.model.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)

        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.model.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(self.model.guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.model.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.model.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
     
            log_new_x_recon = cf_log_x_recon + self.model.guidance_scale * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.model.transformer.zero_vector), dim=1)
            return log_pred

        sample_type = "top"+str(truncation_rate)+add_string

        if len(sample_type.split(',')) > 1: # fast
            if sample_type.split(',')[1][:1]=='q':
                self.model.transformer.p_sample = self.model.p_sample_with_truncation(self.model.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top" and self.model.truncation_forward == False:
            self.model.transformer.cf_predict_start = self.model.predict_start_with_truncation(cf_predict_start, sample_type.split(',')[0])
            self.model.truncation_forward = True
        
        device = "cuda"
        CLIP, _ = clip.load("ViT-B/32", device = device, jit=False)

        mean=(0.48145466, 0.4578275, 0.40821073)
        std=(0.26862954, 0.26130258, 0.27577711)

        if IS_ON_NSML is True:
            data_root = os.path.join(nsml.DATASET_PATH, 'train')
        else:
            data_root = "st1/dataset/coco_vq"

        data = CocoDataset(data_root=data_root, phase='val')
        with torch.no_grad():
            cos_sim = []
            for data_i in torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = False):
                out = self.model.generate_content_for_metric(
                    batch=data_i,
                    filter_ratio=0,
                    sample_type=sample_type
                ) # B x C x H x W

                print(out[0])
                

                # preprocess 256 -> 224
                out = F.interpolate(out, size=(224, 224), mode = 'area') / 255.
                out = torchvision.transforms.Normalize(mean, std)(out)
                img_fts = CLIP.encode_image(out)
                img_fts = F.normalize(img_fts)
                # text preprocess
                text_tokens = clip.tokenize(data_i['text'])['token'].cuda()

                txt_fts = CLIP.encode_text(text_tokens)
                txt_fts = F.normalize(txt_fts)
                sim = F.cosine_similarity(img_fts, txt_fts, dim=-1)
                print("sim", sim)
                cos_sim.append(sim)
            cos_sim = torch.stack(cos_sim, dim=1)
            clip_score = torch.mean(cos_sim)
            print("final", clip_score)

                
    def mask_schedule_test(self, text, truncation_rate, save_root, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, text2=False, purity_temp=1., schedule=0):
        """
        T = 16 fix
        schedule = 1 ~ 4
        1) out -> in
        2) in -> out
        3) grid: blockwise
        4) grid: uniform

        5) random (fast)
        6) purity

        7) random & random revoke
        """
        os.makedirs(save_root, exist_ok=True)

        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = learnable_cf # whether to use learnable classifier-free
        self.model.transformer.prior_rule = prior_rule      # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.model.transformer.prior_weight = prior_weight  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion
        self.model.transformer.purity_temp = purity_temp
        self.model.transformer.mask_schedule_test = schedule

        data_i = {}
        data_i['text'] = [text]

        composition = False
        if text2 != False:
            data_i['text2'] = [text2]
            composition = True
        data_i['image'] = None
        condition = text

        if not IS_ON_NSML:
            os.makedirs(save_root, exist_ok=True)
            str_cond = str(condition)
            save_root_ = os.path.join(save_root, str_cond)
            os.makedirs(save_root_, exist_ok=True)

        if infer_speed != False:
            add_string = 'r,time'+str(infer_speed)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
                composition=composition # text2 condition yes or no
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            wandb.log({"result" : wandb.Image(im)})
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im.save(save_path)

        # recon log step by step
        for i in range(16):
            content = model_out[f"{i}_step_token"]
            content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
            for b in range(content.shape[0]):
                save_base_name = str(i) + '{}'.format(str(cnt).zfill(6))
                im = Image.fromarray(content[b])
                wandb.log({f"{i:02d}_step recon" : wandb.Image(im)})


    def recon_test(self, img_root):
        save_root = os.path.join(img_root, 'recon')
        os.makedirs(save_root, exist_ok=True)
        input_images_list = glob.glob(img_root + '/*.jpg')
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor()
        ])

        imgs = torch.tensor([])
        for f in input_images_list:
            img = Image.open(f)
            img = preprocess(img).unsqueeze(0)
            imgs = torch.cat((imgs, img), 0)

        content = self.model.reconstruct(imgs) # torch.tensor(b, img)
        # content = recon['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)


    def mask_recon_test(self, text, truncation_rate, img_root, batch_size, noise_t, recon_step, guidance_scale=1.0, prior_rule=0, prior_weight=0): 
        """
        input
        =============================================
        text: "~"
        img_root: coco image location
        noise_t: t forward step (1 ~ 100)
            initial token -> fixed, all batch same
        recon_step: recon을 몇 step?
        =============================================

        1) image input(root) -> image
        2) image tokenizing
        3) image masking(handmade? random?) -> 일단 random으로 구현, mask_num
        # mask_num에 따른 t는 dalle에서 연산? ->  O
        # 몇번 iterative하게 sampling할 것 인지는 setting? 

        study:
        1) 몇개 구멍 뚫고, one step recon & 여러번 step으로 recon (기존 sampling에다가 initial condition으로만)
        2) 중요 부분 많이 뚫어서 -> step?
        3) 실제 recon을 내가 손대서 quality 높여볼까?

        self.model.transformer.q_pred(log_x_start, t) => 기존처럼 sampling (given initial condition)
        

        """
        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = True

        save_root = os.path.join(os.path.dirname(img_root), 'recon', f"{text}_{noise_t}_{recon_step}")
        os.makedirs(save_root, exist_ok=True)
        data_i = {}
        data_i['text'] = [text]
        
        condition = text

        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor()
        ])
        img = Image.open(img_root)
        img = preprocess(img).unsqueeze(0) # 1, 3, 256, 256

        data_i['image'] = img

        with torch.no_grad():
            model_out = self.model.mask_recon(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                truncation_rate=truncation_rate,
                noise_t = noise_t,
                recon_step = recon_step
            ) 

        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root, save_base_name +'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)

        masked_index = model_out['masked_index'].to('cpu') # [masked_num]
        recon_token = model_out['recon_token'].squeeze().to('cpu') # 3, 256, 256

        recon_token = recon_token.permute(1, 2, 0).numpy().astype(np.uint8)
        im = Image.fromarray(recon_token)

        # recon img
        recon_path = os.path.join(save_root, 'recon.png')
        im.save(recon_path)

        # recon img with mask
        draw = ImageDraw.Draw(im)        
        for idx in masked_index:
            x, y = divmod(int(idx), 32)
            draw.rectangle((y*8, x*8, y*8+8, x*8+8), outline = 'black', fill = 'black')
        masked_recon_path = os.path.join(save_root, 'masked_recon.png')
        im.save(masked_recon_path)


    def generate_sample_with_replacement(self, text, truncation_rate, save_root, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True):
        os.makedirs(save_root, exist_ok=True)

        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = learnable_cf # whether to use learnable classifier-free
        self.model.transformer.prior_rule = prior_rule      # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.model.transformer.prior_weight = prior_weight  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion

        data_i = {}
        data_i['text'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        if infer_speed != False:
            add_string = 'r,time'+str(infer_speed)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)


    def real_mask_recon(self, data_i, truncation_rate, guidance_scale=5.0): 
        """
        data_i = {'image': b, c, h, w, 'text': b, ~}
        """
        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = True

        with torch.no_grad():
            model_out = self.model.real_mask_return(
                batch=data_i,
                filter_ratio=0,
                content_ratio=1,
                return_att_weight=False,
                truncation_rate=truncation_rate,
            ) # {'t', 'changed', 'recon_token'}

        return model_out

class VQ_Critic(nn.Module):
    def __init__(self, vq='coco', tc_config='configs/token_critic.yaml', tc_learnable_cf=True):
        super().__init__()
        self.VQ_Diffusion = VQ_Diffusion(config='configs/coco_tune.yaml')

        tc_config = load_yaml_config(tc_config)
        
        self.Token_Critic = Token_Critic(config=tc_config, learnable_cf=tc_learnable_cf)
        self.tc_learnable = tc_learnable_cf
        self.device = "cuda"
        self.num_classes = self.VQ_Diffusion.model.transformer.num_classes

    def load_tc(self, ckpt_path):
        bind_model(0, 0, self.Token_Critic, None, None, None, 0, False, False)
        resume_info = ckpt_path.rsplit('/', 1)
        session_name = resume_info[0]
        checkpoint_num = resume_info[1]
        nsml.load(checkpoint=checkpoint_num, session=session_name)

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

    def recon_image_log(self, out_idx): # out_idx (b, 1024)
        content = self.VQ_Diffusion.model.content_codec.decode(out_idx)
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            im = Image.fromarray(content[b])
            wandb.log({f"{b:02d}_step recon" : wandb.Image(im)})

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

    @torch.no_grad()
    def inference_generate_sample_with_condition(self, text, batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log="s,t,r", a=0.1, b=0.2):
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
            tc_cf_cond_emb = self.Token_Critic.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
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
        with torch.no_grad():
            for i, (n, diffusion_index) in tqdm(enumerate(zip(n_sample[:-1], time_list[:-1]))): # before last step
                # 1) VQ: 1 step reconstruction
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                _, log_x_recon = self.VQ_Diffusion.model.transformer.p_pred(log_z, cond_emb, t)
                out = self.VQ_Diffusion.model.transformer.log_sample_categorical(log_x_recon) # recon -> sample -> x_0
                out_idx = log_onehot_to_index(out) # b, 1024
                out2_idx = torch.full_like(out_idx, self.num_classes-1).to(out_idx.device) # all mask index list

                # 1.5) VQ: attn map
                attn_map = self.VQ_Diffusion.model.transformer.transformer.att_total

                # 2) TC: Masking based on score
                t_1 = torch.full((batch_size,), time_list[i+1], device=device, dtype=torch.long) # t-1 step
                score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=cond_emb) # b, 1024
                if tc_guidance != None:
                    cf_score = self.Token_Critic.inference_score(input={'t': t_1, 'recon_token': out_idx}, cond_emb=tc_cf_cond_emb)
                    score = cf_score + tc_guidance * (score - cf_score)
                score = 1 - score # (1 - score) = unchanged(sample) confidence score (because score means changed confidence)

                score += self.n_t(diffusion_index, a, b) # n(t) for randomness
                # score *= 2 # weight sampling

                for ii in range(batch_size):
                    sel = torch.multinomial(score[ii], n)
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




if __name__ == '__main__':
    model = VQ_Critic(vq='coco', tc_config='configs/token_critic.yaml', tc_learnable_cf=True)
    model.load_tc('ailab002/kaist_coco_vq/365/29')

    # wandb.init(project='Att test', name = 'teddy bear attn')
    model.inference_generate_sample_with_condition(text="A picture of a teddy bear on a stone.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='s,r,t', a=0, b=0) # s: score, r: recon image, t: token matrix
    # wandb.finish()

    # wandb.init(project='Att test', name = 'cat attn')
    model.inference_generate_sample_with_condition(text="A cat laying on a computer desk next to a laptop.", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='s,r,t', a=0, b=0) # s: score, r: recon image, t: token matrix
    # wandb.finish()

    # wandb.init(project='Att test', name = 'bear attn')
    model.inference_generate_sample_with_condition(text="Teddy bear playing in the pool", batch_size=4, vq_guidance=5.0, vq_tr=0.86, tc_guidance=None, step=16, wandb_log='s,r,t', a=0, b=0) # s: score, r: recon image, t: token matrix
    # wandb.finish()

