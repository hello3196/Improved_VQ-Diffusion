# Improved_VQ-Diffusion

#TBD

## mask schedule test
ex)

import wandb

wandb.init(project='mask_test', name = '~')

VQ_Diffusion_model.mask_schedule_test("A photo of a confused grizzly bear in calculus class ", truncation_rate=0.86, save_root="exp/mask_schedule_test/grid_uniform", batch_size=4, guidance_scale = 5.0, schedule=4)

schedule = 1 ~ 4 (1: out->in, 2: in->out, 3: blockwise, 4: parallel)
