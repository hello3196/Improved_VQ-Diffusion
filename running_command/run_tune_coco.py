import os
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

if IS_ON_NSML:
    string = "python train.py --name coco_tune --config_file configs/coco_tune.yaml --num_node 1 --tensorboard --load_path " + diffusion_model_path
else:
    string = "python train.py --name coco_tune --config_file configs/coco_tune.yaml --num_node 1 --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained.pth"

os.system(string)

