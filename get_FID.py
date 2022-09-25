import torch
import numpy as np
import dnnlib
from image_synthesis.data.mscoco_dataset import CocoDataset 

# def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
#     """Download the given URL and return a binary-mode file object to access the data."""
#     assert num_attempts >= 1
#     assert not (return_filename and (not cache))

#     if not re.match('^[a-z]+://', url):
#         return url if return_filename else open(url, "rb")

#     if url.startswith('file://'):
#         filename = urllib.parse.urlparse(url).path
#         if re.match(r'^/[a-zA-Z]:', filename):
#             filename = filename[1:]
#         return filename if return_filename else open(filename, "rb")

#     assert is_url(url)

#     # Lookup from cache.
#     if cache_dir is None:
#         cache_dir = make_cache_dir_path('downloads')

#     url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
#     if cache:
#         cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
#         if len(cache_files) == 1:
#             filename = cache_files[0]
#             return filename if return_filename else open(filename, "rb")

#     # Download.
#     url_name = None
#     url_data = None
#     with requests.Session() as session:
#         if verbose:
#             print("Downloading %s ..." % url, end="", flush=True)
#         for attempts_left in reversed(range(num_attempts)):
#             try:
#                 with session.get(url) as res:
#                     res.raise_for_status()
#                     if len(res.content) == 0:
#                         raise IOError("No data received")

#                     if len(res.content) < 8192:
#                         content_str = res.content.decode("utf-8")
#                         if "download_warning" in res.headers.get("Set-Cookie", ""):
#                             links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
#                             if len(links) == 1:
#                                 url = requests.compat.urljoin(url, links[0])
#                                 raise IOError("Google Drive virus checker nag")
#                         if "Google Drive - Quota exceeded" in content_str:
#                             raise IOError("Google Drive download quota exceeded -- please try again later")

#                     match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
#                     url_name = match[1] if match else url
#                     url_data = res.content
#                     if verbose:
#                         print(" done")
#                     break
#             except KeyboardInterrupt:
#                 raise
#             except:
#                 if not attempts_left:
#                     if verbose:
#                         print(" failed")
#                     raise
#                 if verbose:
#                     print(".", end="", flush=True)

#     # Save to cache.
#     if cache:
#         safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
#         cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
#         temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
#         os.makedirs(cache_dir, exist_ok=True)
#         with open(temp_file, "wb") as f:
#             f.write(url_data)
#         os.replace(temp_file, cache_file) # atomic
#         if return_filename:
#             return cache_file

#     # Return data as file object.
#     assert not return_filename
#     return io.BytesIO(url_data)

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
    # progress = progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0)
    
    with torch.no_grad():
        for data_i in torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = False):
            features = detector(data_i["image"].to(device), **detector_kwargs)
            stats.append_torch(features, num_gpus=1, rank=0)
            # progress.update(stats.num_items)
            del data_i
    return stats

def get_gen_FID(data_root, model, detector_url, detector_kwargs, device, batch_size, sample_type, rel_lo=0, rel_hi=1, **stats_kwargs):
    data = CocoDataset(data_root=data_root, phase='val')

    num_items = len(data)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = progress.sub(tag='generator features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0, verbose=progress.verbose)

    with torch.no_grad():
        for data_i in torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle = False):
            out = model.generate_content_for_metric(
                batch=data_i,
                filter_ratio=0,
                sample_type=sample_type
            ) # B x C x H x W

            features = detector(out, **detector_kwargs)
            stats.append_torch(features, num_gpus=1, rank=0)
            progress.update(stats.num_items)
            del out, data_i
    return stats
