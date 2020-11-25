import numpy as np
import torch
import cv2
import os
from torch.utils.data import (
    Dataset,
    DataLoader
)


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


# array(320,320,3)---> tensor(3,320,320)
def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)].astype(np.float32)  # bgr--> rgb
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


# tensor(3,320,320) ---> array(320,320,3)
def torch2cv(t_img):
    t_img = t_img.numpy()
    t_img = t_img[0,:,:,:]
    # t_img = t_img.squeeze(0)
    return t_img.swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]  # rgb--> bgr


def transforms(hdr):
    hdr = cv2.resize(hdr, (320, 320))
    # maxvalue1 = np.max(hdr)
    # hdr = hdr / maxvalue1  # 归一化，将像素值映射到[0,1]
    hdr = map_range(hdr, 0, 1)    # black
    hdr = cv2torch(hdr)  # (3,320,320) tensor
    return hdr


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


class DirectoryDataset(Dataset):
    def __init__(
            self,
            cover_path='images/train',
            secret_path='secret_img',
            data_extensions=['.hdr', '.exr'],
            preprocess=None
    ):
        super(DirectoryDataset, self).__init__()

        cover_path = process_path(cover_path)
        self.cover_list = []
        for root, _, fnames in sorted(os.walk(cover_path)):
            for fname in fnames:
                if any(
                        fname.lower().endswith(extension)
                        for extension in data_extensions
                ):
                    self.cover_list.append(os.path.join(root, fname))
        if len(self.cover_list) == 0:
            msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
            raise RuntimeError(
                msg.format(', '.join(data_extensions), cover_path)
            )

        secret_path = process_path(secret_path)
        self.secret_list = []
        for root, _, fnames in sorted(os.walk(secret_path)):
            for fname in fnames:
                if any(
                        fname.lower().endswith(extension)
                        for extension in data_extensions
                ):
                    self.secret_list.append(os.path.join(root, fname))
        if len(self.secret_list) == 0:
            msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
            raise RuntimeError(
                msg.format(', '.join(data_extensions), cover_path)
            )

        self.preprocess = preprocess

    def __getitem__(self, index):
        cover = cv2.imread(
            self.cover_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        )

        secret = cv2.imread(
            self.secret_list[0], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
            # TODO: 秘密图像可以自己选择是哪一张
        )

        if self.preprocess is not None:
            cover = self.preprocess(cover)
            secret = self.preprocess(secret)

        img_6ch = torch.cat([cover, secret], dim=0)

        return (img_6ch, cover, secret)

    def __len__(self):
        return len(self.cover_list)


image_dataset = DirectoryDataset(preprocess=transforms)

dataloader = DataLoader(image_dataset, batch_size=4, num_workers=0, shuffle=True, drop_last=True)

for i, (concat_img, cover_img, secret_img) in enumerate(dataloader):
    cover_img = torch2cv(cover_img)
    secret_img = torch2cv(secret_img)
    cover_name = './qy/cover.hdr'
    secret_name = './qy/secret.hdr'

    # tone map  ldr
    Du=cv2.createTonemapDurand(2)
    cover_img_ldr=Du.process(cover_img)
    cover_img_ldr=np.clip(cover_img_ldr,0,1)

    secret_img_ldr = Du.process(secret_img)
    secret_img_ldr = np.clip(secret_img_ldr,0,1)

    vtich = np.vstack((cover_img_ldr, secret_img_ldr))
    name = './qy/two.jpg'

    cv2.imwrite(cover_name, cover_img)
    cv2.imwrite(cover_name+'.jpg', (cover_img_ldr*255).astype(int))
    cv2.imwrite(secret_name+'.jpg', (secret_img_ldr*255).astype(int))
    cv2.imwrite(name, (vtich * 255).astype(int))

    if i == 1:
        break


# htich = np.hstack((img,blur2))
# htich2 = np.hstack((blur3,blur4))
# vtich = np.vstack((htich, htich2))