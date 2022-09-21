# Improved_VQ-Diffusion

#TBD

## mask schedule test
ex)

import wandb

wandb.init(project='mask_test', name = '~')

VQ_Diffusion_model.mask_schedule_test("A photo of a confused grizzly bear in calculus class ", truncation_rate=0.86, save_root="exp/mask_schedule_test/grid_uniform", batch_size=4, guidance_scale = 5.0, schedule=4)

schedule = 1 ~ 4 (1: out->in, 2: in->out, 3: blockwise, 4: parallel)


## Token Critic test

### 1) Real Image -> random masking -> recon test
example)

from inference_VQ_Diffusion import VQ_Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='configs/ithq.yaml', path='OUTPUT/pretrained_model/ithq_learnable.pth')

VQ_Diffusion_model.mask_recon_test(text="A person holding a napkin and eating a hotdog", truncation_rate=0.86, img_root="recon_test2/COCO_val2014_000000003310.jpg", batch_size=4, noise_t=50, recon_step=10, guidance_scale=5.0, )
