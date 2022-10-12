import os
import visdom
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info
import torch

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
    def load(filename, map_location=None, **kwargs):
        ckpt = torch.load(filename, map_location=map_location)
        return ckpt

    def save(filename, state_dict, **kwargs):
        save_path = os.path.join(filename, 'model.pkl')
        torch.save(state_dict, save_path)
        return save_path

    if IS_ON_NSML is True:
        nsml.bind(save=save, load=load)
