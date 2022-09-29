# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
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
from image_synthesis.utils.misc import get_model_parameters_info
import image_synthesis.modeling.modules.clip.clip as clip
from image_synthesis.data.mscoco_dataset import CocoDataset 

try:
    import nsml
    from nsml import IS_ON_NSML
    from nsml_utils import bind_model, Logger
    data = os.path.join(nsml.DATASET_PATH[0], 'train')
    clip_model_path = os.path.join(nsml.DATASET_PATH[1], 'train/ViT-B-32.pt')
    diffusion_model_path = os.path.join(nsml.DATASET_PATH[2], 'train/ithq_learnable.pth')
    vqvae_model_path = os.path.join(nsml.DATASET_PATH[3], 'train/ithq_vqvae.pth')
except ImportError:
    nsml = None
    IS_ON_NSML = False

import wandb

resume_nsml_model = False

class VQ_Diffusion():
    def __init__(self, config, path, imagenet_cf=False):
        if IS_ON_NSML:
            bind_model()
            self.nsml_img_logger = Logger()
        if resume_nsml_model:
            self.info = self.get_nsml_model(ema=True, model_path=path, config_path=config, imagenet_cf=imagenet_cf)
        else:
            self.info = self.get_model(ema=True, model_path=path, config_path=config, imagenet_cf=imagenet_cf)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad=False

    def get_nsml_model(self, ema, model_path, config_path, imagenet_cf):
        resume_info = model_path.split(',')
        checkpoint_num = resume_info[0]
        session_name = resume_info[1]
        print(f'Resuming from checkpoint {checkpoint_num}, session {session_name}')
        return nsml.load(checkpoint=checkpoint_num, session=session_name, map_location="cpu")

    def get_model(self, ema, model_path, config_path, imagenet_cf):
        # if 'OUTPUT' in model_path: # pretrained model
        #     model_name = model_path.split(os.path.sep)[-3]
        # else:
        #     model_name = os.path.basename(config_path).replace('.yaml', '')
        model_name = model_path

        config = load_yaml_config(config_path)

        if imagenet_cf:
            config['model']['params']['diffusion_config']['params']['transformer_config']['params']['class_number'] = 1001

        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
        else:
            print("Model path: {} does not exist.".format(model_path))
            exit(0)

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        if ema==True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
    
        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

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
            if IS_ON_NSML:
                self.nsml_img_logger.images_summary(save_base_name, im)
            else:
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
            if IS_ON_NSML:
                self.nsml_img_logger.images_summary(save_base_name, im)
            else:
                save_path = os.path.join(save_root_, save_base_name+'.png')
                im.save(save_path)
        
        # recon log step by step
        # for i in range(9):
        #     content = model_out[f"{i}_step_token"]
        #     content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        #     for b in range(content.shape[0]):
        #         im = Image.fromarray(content[b])
        #         wandb.log({f"{i}_step recon" : wandb.Image(im)})

    def inference_generate_sample_for_fid(self, truncation_rate, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, text2=False, purity_temp=1., schedule=0):
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

        """
        schedule 수정 (step, schedule = 7 ...)
        """
        if schedule == 7:
            for s in range(1, len(self.model.transformer.n_sample)):
                self.model.transformer.n_sample[s] += self.model.transformer.n_sample[s - 1]
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

        if IS_ON_NSML is True:
            data_root = data
        else:
            data_root = "st1/dataset/coco_vq"

        device = "cuda"
        detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        detector_kwargs = dict(return_features=True)

        mu_real, sigma_real = get_FID.get_real_FID(data_root=data_root, detector_url=detector_url, detector_kwargs=detector_kwargs,
                            device=device, batch_size=batch_size, rel_lo=0, rel_hi=0, capture_mean_cov=True).get_mean_cov
        mu_gen, sigma_gen = get_FID.get_gen_FID(data_root=data_root, model=self.model, detector_url=detector_url, detector_kwargs=detector_kwargs,
                            device=device, batch_size=batch_size, sample_type=sample_type, rel_lo=0, rel_hi=1, capture_mean_cov=True).get_mean_cov

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        print("FID : ", float(fid))

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
            if IS_ON_NSML:
                self.nsml_img_logger.images_summary(save_base_name, im)
            else:
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
                if IS_ON_NSML:
                    self.nsml_img_logger.images_summary(save_base_name, im)


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


if __name__ == '__main__':
    VQ_Diffusion_model = VQ_Diffusion(config='configs/ithq.yaml', path=diffusion_model_path)
    # VQ_Diffusion_model = VQ_Diffusion(config='configs/ithq.yaml', path='OUTPUT/pretrained_model/ithq_learnable.pth')

    # Inference VQ-Diffusion
    # VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=0.86, save_root="RESULT", batch_size=4)

    # Inference Improved VQ-Diffusion with zero-shot classifier-free sampling
    # VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0, learnable_cf=False)
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0, learnable_cf=False)

    # Inference Improved VQ-Diffusion with learnable classifier-free sampling
    # VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0)
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0)

    # Inference Improved VQ-Diffusion for metric
    VQ_Diffusion_model.inference_generate_sample_for_fid(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=5)

    # Inference Improved VQ-Diffusion with fast/high-quality inference
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=0.86, save_root="RESULT", batch_size=4, infer_speed=0.5) # high-quality inference, 0.5x inference speed
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=0.86, save_root="RESULT", batch_size=4, infer_speed=2) # fast inference, 2x inference speed
    # infer_speed shoule be float in [0.1, 10], larger infer_speed means faster inference and smaller infer_speed means slower inference

    # Inference Improved VQ-Diffusion with purity sampling
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=0.86, save_root="RESULT", batch_size=4, prior_rule=2, prior_weight=1) # purity sampling

    # Inference Improved VQ-Diffusion with both learnable classifier-free sampling and fast inference
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0, infer_speed=2) # classifier-free guidance and fast inference




    # VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/coco_learnable.pth')

    # Inference VQ-Diffusion
    # VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=0.86, save_root="RESULT", batch_size=4)

    # Inference Improved VQ-Diffusion with learnable classifier-free sampling
    # VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=3.0)




    # Inference Improved VQ-Diffusion with zero-shot classifier-free sampling: load models without classifier-free fine-tune and set guidance_scale to > 1
    # VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/coco_pretrained.pth')
    # VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=0.86, save_root="RESULT", batch_size=4, guidance_scale=3.0, learnable_cf=False)




    # Inference VQ-Diffusion
    # VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_pretrained.pth')
    # VQ_Diffusion_model.inference_generate_sample_with_class(407, truncation_rate=0.86, save_root="RESULT", batch_size=4)


    # Inference Improved VQ-Diffusion with classifier-free sampling
    # VQ_Diffusion_model = VQ_Diffusion(config='configs/imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_learnable.pth', imagenet_cf=True)
    # VQ_Diffusion_model.inference_generate_sample_with_class(407, truncation_rate=0.94, save_root="RESULT", batch_size=4, guidance_scale=1.5)
