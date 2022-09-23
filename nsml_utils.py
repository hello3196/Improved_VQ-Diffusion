import os
import visdom
import wget
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info

try:
    import nsml
    from nsml import IS_ON_NSML
except ImportError:
    nsml = None
    IS_ON_NSML = False

class Logger(object):
    def __init__(self,):
        self.last = None
        self.viz = nsml.Visdom(visdom=visdom)
    
    def scalar_summary(self, tag, value, step):
        if self.last and self.last['step'] != step:
            nsml.report(**self.last)
            self.last = None
        if self.last is None:
            self.last = {
                'step': step, 
                'iter': step, 
                'epoch': 1
            }
        self.last[tag] = value

    def images_summary(self, tag, images, step=0):
        self.viz.image(
            images, 
            opts = dict(title=tag, caption='%s/%d' % (tag, step)))

    def histo_summary(self, tag, values, step, bins=1000):
        pass


def bind_model():
    def load(filename, config_path, imagenet_cf, ema, **kwargs):
        model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)
        if imagenet_cf:
            config['model']['params']['diffusion_config']['params']['transformer_config']['params']['class_number'] = 1001
        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        print(model_parameters)
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location="cpu")
        else:
            print("Model path: {} does not exist.".format(filename))
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

    def save(filename, **kwargs):
        wget.download("https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/ViT-B-32.pt?sv=2019-12-12&st=2022-03-09T01%3A57%3A52Z&se=2028-04-10T01%3A57%3A00Z&sr=b&sp=r&sig=bj5P0BbkreoGdbjDK4sZ5tis%2BwltrVAiN9DQdmzHpEE%3D")
        wget.download("https://facevcstandard.blob.core.windows.net/v-zhictang/Improved-VQ-Diffusion_model_release/coco_learnable.pth?sv=2020-10-02&st=2022-05-30T10%3A21%3A22Z&se=2030-05-31T10%3A21%3A00Z&sr=b&sp=r&sig=nhTx1%2B6rK6hWR9CVGuPauKnamayHXfDu1E8RGD5%2FRw0%3D")
        #wget.download("")

    def infer(input):
        stats_metrics = dict()
        snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
        for metric in metrics:
            result_dict = metric_main.calc_metric(metric=metric, G=named_models['G_ema'], D=named_models['D'],
                dataset_kwargs=training_set_kwargs, testset_kwargs=testing_set_kwargs, num_gpus=num_gpus, rank=rank, device=device, txt_recon=True, img_recon=False, metric_only_test=metric_only_test)
            if rank == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
            stats_metrics.update(result_dict.results)
        return stats_metrics

    if IS_ON_NSML is True:
        nsml.bind(save=save, load=load, infer=infer)
