import numpy as np
import torch
import cv2
import torchvision.utils as vutils


# import skimage.io as io

def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)

# array(320,320,3)---> tensor(320,320,3)
def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)].astype(np.float32)
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))

# tensor(320,320,3) ---> array(3,320,320)
def torch2cv(t_img):
    return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]

def transforms(hdr):
    # b, g, r = cv2.split(hdr)
    # hdr = cv2.merge([r, g, b])
    hdr = cv2.resize(hdr, (320, 320))
    maxvalue1 = np.max(hdr)
    hdr = hdr / maxvalue1  # 归一化，将像素值映射到[0,1]
    # hdr = map_range(hdr, 0, 1)    # black
    hdr = cv2torch(hdr)  # (3,320,320)
    return hdr


def save_pic(phase, cover, stego, secret, secret_rev, save_path, batch_size):
    showContainer = torch.cat([cover, stego], 2)
    # print(showContainer)
    showReveal = torch.cat([secret, secret_rev], 2)
    resultImg = torch.cat([showContainer, showReveal], 2)
    # resultImg = torch2cv(torch.cat([showContainer, showReveal], 2)).detach().cpu()
    # print(resultImg.size())
    if phase == 'train':
        resultImgName = '%s/11fbfbg.jpg' % save_path
    vutils.save_image(resultImg, resultImgName, nrow=batch_size, padding=1, normalize=False)

import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.RGBE import readHdr, rgbe2float


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
        # pdb.set_trace()
        cover = cv2.imread(
            self.cover_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        )

        test = rgbe2float(readHdr(self.cover_list[index]))
        # cover = cv2.resize(cover, (320, 320))
        # cover = cover/255
        # cover = cv2torch(cover)

        secret = cv2.imread(
            self.secret_list[0], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
            # TODO: 秘密图像可以自己选择是哪一张
        )
        # secret = cv2.resize(secret, (320, 320))
        # secret = secret/255
        # secret = cv2torch(secret)

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
    cover_img
    secret_img
    save_pic('train', cover_img, cover_img, secret_img, secret_img, './test_pics', 4)
    vutils.save_image(cover_img, './test_pics/qqqqq.jpg', nrow=1, padding=1, normalize=False)
    vutils.save_image(secret_img, './test_pics/sss.jpg', nrow=1, padding=1, normalize=False)

    if i == 1:
        break


# 用PIL读出来的图像是一个对象，不是array格式，许哟啊通过np.array()转换为array,才能在plt上展示，但是可以直接show()
# 用cv2读出来的图是array格式
# img1 = cv2.imread('./images/train/3DTotal_free_sample_1_Ref.hdr')
# # img2 = cv2.imread('./training/train/resultPics/trainResultPics_epoch0000.jpg')
# img2 = transforms(img1)
# img2 = img2.cuda()
# vutils.save_image(img2, './test_pics/2.jpg', nrow=1, padding=1, normalize=False)
#
# img3 = torch2cv(img2.cpu())
# img4 = cv2torch(img1)
# vutils.save_image(img4, './test_pics/3.jpg', nrow=1, padding=1, normalize=False)

# fig = plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(img1)
# plt.subplot(2,2,2)
# plt.imshow(img3)
# plt.subplot(2,2,3)
# plt.imshow(img3)
# plt.subplot(2,2,4)
# plt.imshow(img3)
# plt.show()