import os
import numpy as np
from numpy.random import uniform
import torch
from torch.utils.data import Dataset
import cv2
import pdb
#返回绝对路径
def process_path(directory, create=False):
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    directory = os.path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory

#返回目录名 文件名 扩展名
def split_path(directory):
    directory = process_path(directory)
    name, ext = os.path.splitext(os.path.basename(directory))
    return os.path.dirname(directory), name, ext


# From torchnet
# https://github.com/pytorch/tnt/blob/master/torchnet/transform.py
def compose(transforms):
    'Composes list of transforms (each accept and return one item)'
    assert isinstance(transforms, list)
    for transform in transforms:
        assert callable(transform), 'list of functions expected'

    def composition(obj):
        'Composite function'
        for transform in transforms:
            obj = transform(obj)
        return obj

    return composition


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def str2bool(x):
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)].astype(np.float32)
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))

# TODO: daimayouhuas
def torch2cv(t_img):
    t_img = t_img.detach().cpu().numpy()
    t_img = t_img[0, :, :, :]
    # t_img = t_img.squeeze(0)
    return t_img.swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]


def resize(x, size):
    return cv2.resize(x, size)


class Exposure(object):
    def __init__(self, stops=0.0, gamma=1.0):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img * (2 ** self.stops), 0, 1) ** self.gamma

    def __call__(self, img):
        return np.clip(img * (2 ** self.stops), 0, 1) ** self.gamma      


class PercentileExposure(object):
    def __init__(self, gamma=2.0, low_perc=10, high_perc=90, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            low_perc = uniform(0, 15)
            high_perc = uniform(85, 100)
        self.gamma = gamma
        self.low_perc = low_perc
        self.high_perc = high_perc

    def __call__(self, x):
        low, high = np.percentile(x, (self.low_perc, self.high_perc))
        return map_range(np.clip(x, low, high)) ** (1 / self.gamma)


class BaseTMO(object):
    def __call__(self, img):
        return self.op.process(img)


class Reinhard(BaseTMO):
    def __init__(
        self,
        intensity=-1.0,
        light_adapt=0.8,
        color_adapt=0.0,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


class Mantiuk(BaseTMO):
    def __init__(self, saturation=1.0, scale=0.75, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            scale = uniform(0.65, 0.85)

        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )


class Drago(BaseTMO):
    def __init__(self, saturation=1.0, bias=0.85, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            bias = uniform(0.7, 0.9)

        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )


class Durand(BaseTMO):
    def __init__(
        self,
        contrast=3,
        saturation=1.0,
        sigma_space=8,
        sigma_color=0.4,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            contrast = uniform(3.5)
        # self.op = cv2.createTonemapDurand(
        #     contrast=contrast,
        #     saturation=saturation,
        #     sigma_space=sigma_space,
        #     sigma_color=sigma_color,
        #     gamma=gamma,
        # )


TMO_DICT = {
    'exposure': Exposure,
    'reinhard': Reinhard,
    'mantiuk': Mantiuk,
    'drago': Drago,
    'durand': Durand,
}


def tone_map(img, tmo_name, **kwargs):
    return TMO_DICT[tmo_name](**kwargs)(img)


TRAIN_TMO_DICT = {
    #'exposure': PercentileExposure,
    'reinhard': Reinhard,
    'mantiuk': Mantiuk,
    'drago': Drago,
    'durand': Durand,
}


def random_tone_map(x):
    tmos = list(TRAIN_TMO_DICT.keys())
    choice = np.random.randint(0, len(tmos))
    tmo = TRAIN_TMO_DICT[tmos[choice]](randomize=True)
    return map_range(tmo(x))


def hdr2ldr(hdr):
    Du = cv2.createTonemapDurand(2)
    ldr = Du.process(hdr)
    ldr = np.clip(ldr, 0, 1)
    return ldr


def create_tmo_param_from_args(opt):
    if opt.tmo == 'exposure':
        return {k: opt.get(k) for k in ['gamma', 'stops']}
    else:  # TODO: Implement for others
        return {}


def clamped_gaussian(mean, std, min_value, max_value):
    if max_value <= min_value:
        return mean
    factor = 0.99
    while True:
        ret = np.random.normal(mean, std)
        if ret > min_value and ret < max_value:
            break
        else:
            std = std * factor
            ret = np.random.normal(mean, std)

    return ret


def exponential_size(val):
    return val * (np.exp(-np.random.uniform())) / (np.exp(0) + 1)


# Accepts hwc-bgr image
def index_gauss(
    img,
    precision=None,
    crop_size=None,
    random_size=True,
    ratio=None,
    seed=None,
):
    """Returns indices (Numpy slice) of an image crop sampled spatially using a gaussian distribution.

    Args:
        img (Array): Image as a Numpy array (OpenCV view, hwc-BGR).
        precision (list or tuple, optional): Floats representing the precision
            of the Gaussians (default [1, 4])
        crop_size (list or tuple, optional): Ints representing the crop size
            (default [img_width/4, img_height/4]).
        random_size (bool, optional): If true, randomizes the crop size with
            a minimum of crop_size. It uses an exponential distribution such
            that smaller crops are more likely (default True).
        ratio (float, optional): Keep a constant crop ratio width/height (default None).
        seed (float, optional): Set a seed for np.random.seed() (default None)

    Note:
        - If `ratio` is None then the resulting ratio can be anything.
        - If `random_size` is False and `ratio` is not None, the largest dimension
          dictated by the ratio is adjusted accordingly:
                
                - `crop_size` is (w=100, h=10) and `ratio` = 9 ==> (w=90, h=10)
                - `crop_size` is (w=100, h=10) and `ratio` = 0.2 ==> (w=100, h=20)

    """
    np.random.seed(seed)
    dims = {'w': img.shape[1], 'h': img.shape[0]}
    if precision is None:
        precision = {'w': 1, 'h': 4}
    else:
        precision = {'w': precision[0], 'h': precision[1]}

    if crop_size is None:
        crop_size = {key: int(dims[key] / 4) for key in dims}
    else:
        crop_size = {'w': crop_size[0], 'h': crop_size[1]}

    if ratio is not None:
        ratio = max(ratio, 1e-4)
        if ratio > 1:
            if random_size:
                crop_size['h'] = int(
                    max(crop_size['h'], exponential_size(dims['h']))
                )
            crop_size['w'] = int(np.round(crop_size['h'] * ratio))
        else:
            if random_size:
                crop_size['w'] = int(
                    max(crop_size['w'], exponential_size(dims['w']))
                )
            crop_size['h'] = int(np.round(crop_size['w'] / ratio))
    else:
        if random_size:
            crop_size = {
                key: int(max(val, exponential_size(dims[key])))
                for key, val in crop_size.items()
            }

    centers = {
        
        key: int(
            clamped_gaussian(
                dim / 2,
                crop_size[key] / precision[key],
                min(int(crop_size[key] / 2), dim),
                max(int(dim - crop_size[key] / 2), 0),
            )
        )
        for key, dim in dims.items()
    }
    starts = {
        key: max(center - int(crop_size[key] / 2), 0)
        for key, center in centers.items()
    }
    ends = {key: start + crop_size[key] for key, start in starts.items()}
    return np.s_[starts['h'] : ends['h'], starts['w'] : ends['w'], :]


def slice_gauss(
    img,
    precision=None,
    crop_size=None,
    random_size=True,
    ratio=None,
    seed=None,
):
    """Returns a cropped sample from an image array using :func:`index_gauss`"""
    return img[index_gauss(img, precision, crop_size, random_size, ratio)]


class DirectoryDataset(Dataset):
    def __init__(
        self,
        image_path='hdr/train',
        data_extensions=['.hdr', '.exr', '.jpeg'],
        preprocess=None,
    ):
        super(DirectoryDataset, self).__init__()
        
        image_path = process_path(image_path)
        self.image_list = []

        for root, _, fnames in sorted(os.walk(image_path)):
            for fname in fnames:
                if any(
                    fname.lower().endswith(extension)
                    for extension in data_extensions
                ):
                    self.image_list.append(os.path.join(root, fname))
        if len(image_path) == 0:
            msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
            raise RuntimeError(
                msg.format(', '.join(data_extensions), image_path)
            )

        # cover_path = process_path(cover_path)
        # self.cover_list = []
        # for root, _, fnames in sorted(os.walk(cover_path)):
        #     for fname in fnames:
        #         if any(
        #             fname.lower().endswith(extension)
        #             for extension in data_extensions
        #         ):
        #             cover_list.append(os.path.join(root, fname))
        # if len(cover_path) == 0:
        #     msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
        #     raise RuntimeError(
        #         msg.format(', '.join(data_extensions), cover_path)
        #     )
        #
        #
        # secret_path = process_path(secret_path)
        # self.secret_list = []
        # for root, _, fnames in sorted(os.walk(secret_path)):
        #     for fname in fnames:
        #         if any(
        #                 fname.lower().endswith(extension)
        #                 for extension in data_extensions
        #         ):
        #             self.secret_list.append(os.path.join(root, fname))
        # # pdb.set_trace()
        # if len(self.secret_list) == 0:
        #     msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
        #     raise RuntimeError(
        #         msg.format(', '.join(data_extensions), cover_path)
        #     )

        # self.cover_list = image_list[0: int(len(image_list)/2)]
        # self.secret_list = image_list[int(len(image_list) / 2): len(image_list)]

        self.preprocess = preprocess

    def __getitem__(self, index):
        img = cv2.imread(self.image_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        # cover = cv2.imread(
        #     self.cover_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        # )
        #
        # secret = cv2.imread(
        #     self.secret_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        # )

        if self.preprocess is not None:
              img = self.preprocess(img)
        #     cover = self.preprocess(cover)
        #     secret = self.preprocess(secret)
        #
        # img_6ch = torch.cat([cover, secret], dim=0)

        # return (img_6ch, cover, secret)
        return img

    def __len__(self):
        # return len(self.cover_list)
        return len(self.image_list)