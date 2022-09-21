import os
import dnnlib
import legacy
import copy
import pickle
from metrics import metric_main
from torch_utils import misc
import visdom

try:
    import nsml
    from nsml import IS_ON_NSML
except ImportError:
    nsml = None
    IS_ON_NSML = False

class Logger(object):
    def __init__(self, log_dir):
        print('log dir: ', log_dir)
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

    def images_summary(self, tag, images, step):
        self.viz.image(
            images, 
            opts = dict(title=tag, caption='%s/%d' % (tag, step)))

    def histo_summary(self, tag, values, step, bins=1000):
        pass


def bind_model(named_models, num_gpus, device, rank, cur_nimg, run_dir, training_set_kwargs, testing_set_kwargs, metrics, metric_only_test):
    def load(filename, **kwargs):
        with dnnlib.util.open_url(os.path.join(filename, 'model.pkl')) as f:
            print(f)
            resume_data = legacy.load_network_pkl(f)
        print(resume_data['G'])
        for name, module in named_models:
            if name == 'augment_pipe': continue
            print(name, module)
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        print('Model loaded from {}'.format(os.path.join(filename, 'model.pkl')))

    def save(filename, **kwargs):
        snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
        for name, module in named_models:
            if module is not None:
                if num_gpus > 1:
                    misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            snapshot_data[name] = module
            del module # conserve memory
            
            if rank == 0:
                with open(os.path.join(filename, 'model.pkl'), 'wb') as fp:
                    pickle.dump(snapshot_data, fp)

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
        nsml.bind(save=save, load=load, infer=infer, rank=rank, cur_nimg=cur_nimg)
