import torch
import numpy as np
import dnnlib
from image_synthesis.data.mscoco_dataset import CocoDataset
from image_synthesis.data.build import build_dataloader

_feature_detector_cache = dict()

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        with dnnlib.util.open_url(url, verbose=verbose) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)

    return _feature_detector_cache[key]

def get_real_FID(data_root, detector_url, detector_kwargs, device, batch_size, rel_lo=0, rel_hi=1, **stats_kwargs):
    data = CocoDataset(data_root=data_root, phase='val')
    num_items = len(data)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0)
    with torch.no_grad():
        for data_i in torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = False):
            print(1)
            image = data_i["image"].to(device)
            features = detector(image, **detector_kwargs)
            print(2)
            stats.append_torch(features, num_gpus=1, rank=0)
            print(3)
            del data_i
            print(4)
    return stats

def get_gen_FID(data_root, model, detector_url, detector_kwargs, device, batch_size, sample_type, rel_lo=0, rel_hi=1, **stats_kwargs):
    data = CocoDataset(data_root=data_root, phase='val')
    num_items = len(data)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0)
    with torch.no_grad():
        for data_i in torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = False):
            out = model.generate_content_for_metric(
                batch=data_i,
                filter_ratio=0,
                sample_type=sample_type
            ) # B x C x H x W

            features = detector(out.to(device), **detector_kwargs)
            stats.append_torch(features, num_gpus=1, rank=0)
            del out, data_i
    return stats
