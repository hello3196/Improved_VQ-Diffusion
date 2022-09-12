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

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info
import image_synthesis.modeling.modules.clip.clip as clip
from image_synthesis.data.mscoco_dataset import CocoDataset 

import wandb
class VQ_Diffusion():
    def __init__(self, config, path, imagenet_cf=False):
        self.info = self.get_model(ema=True, model_path=path, config_path=config, imagenet_cf=imagenet_cf)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad=False

    def get_model(self, ema, model_path, config_path, imagenet_cf):
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else: 
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)

        if imagenet_cf:
            config['model']['params']['diffusion_config']['params']['transformer_config']['params']['class_number'] = 1001

        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

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
        os.makedirs(save_root, exist_ok=True)

        self.model.guidance_scale = guidance_scale

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

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
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, text2=False, purity_temp=1., ):
        os.makedirs(save_root, exist_ok=True)

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
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)
        
        # recon log step by step
        # for i in range(9):
        #     content = model_out[f"{i}_step_token"]
        #     content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        #     for b in range(content.shape[0]):
        #         im = Image.fromarray(content[b])
        #         wandb.log({f"{i}_step recon" : wandb.Image(im)})

    def inference_generate_sample_for_metric(self, truncation_rate, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, text2=False, purity_temp=1., schedule=0):
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

            # print(f"[{t}]no condition: ", cf_log_x_recon)
            # print(f"[{t}]condition: ", log_x_recon)            
            log_new_x_recon = cf_log_x_recon + self.model.guidance_scale * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.model.transformer.zero_vector), dim=1)

            # c_mat = (log_x_recon - torch.logsumexp(log_x_recon, dim=1, keepdim=True)).clamp(-70, 0)
            # n_mat = (cf_log_x_recon - torch.logsumexp(cf_log_x_recon, dim=1, keepdim=True)).clamp(-70, 0)

            # c_mat = torch.stack(c_mat.max(dim=1), dim = 1)
            # n_mat = torch.stack(n_mat.max(dim=1), dim = 1)
            
            # for n_m, c_m in zip(n_mat, c_mat):
            #     wandb.log({
            #         'logit_max log(no condition)' : pd.DataFrame(n_m.view(2*32, 32).to('cpu').numpy()),
            #         'logit_max log(condition)' : pd.DataFrame(c_m.view(2*32, 32).to('cpu').numpy()) 
            #     })
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

        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

        data = CocoDataset(data_root="st1/dataset/coco_vq", phase='val')
        with torch.no_grad():
            cos_sim = []
            for data_i in torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = False):
                out = self.model.generate_content_for_metric(
                    batch=data_i,
                    filter_ratio=0,
                    sample_type=sample_type
                ) # B x C x H x W

                # print(out[0])
                # img로 scaling

                # image wandb log
                out_ = out.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                for b in range(out_.shape[0]):
                    im = Image.fromarray(out_[b])
                    wandb.log({f"result" : wandb.Image(im, caption=data_i['text'][b])})

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
                cos_sim.append(sim.to('cpu'))

                if len(cos_sim) >= 50:
                    break
            cos_sim = torch.stack(cos_sim, dim=1)
            clip_score = torch.mean(cos_sim)
            print(schedule,"_schedule final", clip_score)

                
    def mask_schedule_test(self, text, truncation_rate, save_root, batch_size, infer_speed=False, guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, text2=False, purity_temp=1., schedule=0):
        """
        T = 16 fix
        schedule = 1 ~ 4
        1) out -> in
        2) in -> out
        3) grid: blockwise
        4) grid: uniform

        5) random (fast)
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
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im = Image.fromarray(content[b])
            wandb.log({"result" : wandb.Image(im)})
            im.save(save_path)

        # recon log step by step
        for i in range(16):
            content = model_out[f"{i}_step_token"]
            content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
            for b in range(content.shape[0]):
                im = Image.fromarray(content[b])
                wandb.log({f"{i:02d}_step recon" : wandb.Image(im)})

if __name__ == '__main__':
    VQ_Diffusion_model = VQ_Diffusion(config='configs/ithq.yaml', path='OUTPUT/pretrained_model/ithq_learnable.pth')
    # wandb.init(project='clipscore_200', name = 'schedule_2')
    VQ_Diffusion_model.inference_generate_sample_for_metric(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=1)
    # wandb.finish()
    # wandb.init(project='clipscore_200', name = 'schedule_3')
    VQ_Diffusion_model.inference_generate_sample_for_metric(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=2)
    # wandb.finish()
    # wandb.init(project='clipscore_200', name = 'schedule_4')
    VQ_Diffusion_model.inference_generate_sample_for_metric(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=3)
    # wandb.finish()
    # wandb.init(project='clipscore_200', name = 'schedule_5')
    VQ_Diffusion_model.inference_generate_sample_for_metric(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=4)
    # wandb.finish()
    # wandb.init(project='clipscore_200', name = 'schedule_6')
    VQ_Diffusion_model.inference_generate_sample_for_metric(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=5)
    VQ_Diffusion_model.inference_generate_sample_for_metric(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=6)

    # Inference VQ-Diffusion
    # VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=0.86, save_root="RESULT", batch_size=4)

    # Inference Improved VQ-Diffusion with zero-shot classifier-free sampling
    # VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0, learnable_cf=False)
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0, learnable_cf=False)

    # Inference Improved VQ-Diffusion with learnable classifier-free sampling
    # VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0)
    # VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0)

    # Inference Improved VQ-Diffusion for metric
    # VQ_Diffusion_model.inference_generate_sample_for_metric(truncation_rate=1.0, batch_size=4, guidance_scale=5.0, prior_rule=2, prior_weight=1, schedule=1)

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
