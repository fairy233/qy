import argparse
import torch
import time
import os
import cv2
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from models.util import (
    DirectoryDataset,
    cv2torch,
    map_range,
    torch2cv,
    hdr2ldr
)
from models.earlyStop import EarlyStopping
from models.Hnet_unet_res import HNet
from models.RNet import RNet
from models.AverageMeter import AverageMeter
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default="train", help='train | valid | test')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Batch size.'
    )
    parser.add_argument(
        '--image_size', type=int, default=320, help='image size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of data loading workers.',
    )
    parser.add_argument(
        '--images_path', type=str, default='./images/', help='Path to coverImage data.'
    )
    parser.add_argument(
        '--use_gpu', type=bool, default=True, help='Use GPU for training.'
    )
    parser.add_argument(
        '--debug', type=bool, default=False, help='debug mode do not create folders'
    )
    parser.add_argument(
        # 路径可为'./test_pics'
        '--test', default='', help='test mode, you need give the test pics dirs in this param'
    )
    parser.add_argument(
        '--beta', type=float, default=0.5, help='hyper parameter of loss_sum'
    )
    parser.add_argument(
        '--epochs', type=int, default=200, help='the num of training times'
    )
    parser.add_argument(
        '--checkpoint_freq', type=int, default=50, help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '--loss_freq', type=int, default=5, help='Report (average) loss every x epochs.',
    )
    parser.add_argument(
        '--result_freq', type=int, default=10, help='save the resultPictures every x epochs'
    )
    parser.add_argument(
        '--checkpoint_path', default='./training', help='Path for checkpointing.',
    )
    parser.add_argument(
        '--log_path', default='./training', help='log path'
    )
    parser.add_argument(
        '--result_pics', default='./training', help='folder to output training images'
    )
    parser.add_argument(
        '--validation_pics', default='./training', help='folder to output validation images')
    parser.add_argument(
        '--test_pics', default='./test', help='folder to output test images'
    )
    parser.add_argument(
        # './training1025/checkPoints/H_epoch0150_sumloss=0.001211.pth'
        '--Hnet', default='', help="path to Hidenet (to continue training)"
    )
    parser.add_argument(
        # './training1025/checkPoints/R_epoch0150_sumloss=0.001211.pth'
        '--Rnet', default='', help="path to Revealnet (to continue training)"
    )

    return parser.parse_args()


# TODO: 对输入图像的预处理的函数
def transforms(hdr):
    hdr = cv2.resize(hdr, (320, 320))
    # maxvalue1 = np.max(hdr)
    # hdr = hdr / maxvalue1  # 归一化，将像素值映射到[0,1]
    hdr = map_range(hdr, 0, 1)    # black
    hdr = cv2torch(hdr)  # (3,320,320) tensor
    return hdr


def print_log(log_info, log_path, console=True):
    log_info += '\n'
    if console:
        print(log_info)
    # debug mode will not write logs into files
    if not opt.debug:
        # write logs into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:  # 如果地址下有文件，打开这个文件后，再写入日志
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# TODO: 打印网络模型, 这个函数没有调用
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath, console=False)
    print_log('Total number of parameters: %d' % num_params, logPath, console=False)


def save_pic(phase, cover, stego, secret, secret_rev, save_path, batch_size, epoch):
    if not opt.debug:
        # tensor  --> numpy.narray
        cover = torch2cv(cover)
        secret = torch2cv(secret)
        stego = torch2cv(stego)
        secret_rev = torch2cv(secret_rev)

        cover_diff = cover - stego
        secret_diff = secret - secret_rev

        #  np.vstack  np.hstack
        showContainer = np.hstack((cover, stego))
        showReveal = np.hstack((secret, secret_rev))
        resultImg = np.vstack((showContainer, showReveal))

        # tone map  ldr
        cover_ldr = (hdr2ldr(cover) * 255).astype(int)
        secret_ldr = (hdr2ldr(secret) * 255).astype(int)
        stego_ldr = (hdr2ldr(stego) * 255).astype(int)
        secret_rev_ldr = (hdr2ldr(secret_rev) * 255).astype(int)

        cover_diff_ldr = (hdr2ldr(cover_diff) * 255).astype(int)
        secret_diff_ldr = (hdr2ldr(secret_diff) * 255).astype(int)

        showContainer_ldr = np.hstack((cover_ldr, stego_ldr))
        showReveal_ldr = np.hstack((secret_ldr, secret_rev_ldr))
        resultImg_ldr = np.vstack((showContainer_ldr, showReveal_ldr))

        diffImg_ldr = np.vstack((cover_diff_ldr, secret_diff_ldr))

        if phase == 'train':
            resultImgName = '%s/trainResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)
            diffImgName = '%s/train_diff_epoch%04d_batch%02d.jpg' % (save_path, epoch, batch_size)
        if phase == 'valid':
            resultImgName = '%s/valResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)
            diffImgName = '%s/val_diff_epoch%04d_batch%02d.jpg' % (save_path, epoch, batch_size)
        if phase == 'test':
            resultImgName = '%s/testResult.hdr' % save_path
            diffImgName = '%s/test_diff.jpg' % save_path

        # result hdr
        cv2.imwrite(resultImgName, resultImg)
        # result ldr
        cv2.imwrite(resultImgName + '.jpg', resultImg_ldr)
        # diff ldr
        cv2.imwrite(diffImgName, diffImg_ldr)


def train(data_loader, epoch, Hnet, Rnet, criterion):
    start_time = time.time()
    train_Hlosses = AverageMeter()
    train_Rlosses = AverageMeter()
    train_SumLosses = AverageMeter()  # record loss of H-net
    val_Hlosses = AverageMeter()
    val_Rlosses = AverageMeter()
    val_SumLosses = AverageMeter()

    # 早停法
    # patience = 7
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    for phase in ['train', 'valid']:
        if phase == 'train':
            Hnet.train()  # 训练
            Rnet.train()
        else:
            Hnet.eval()  # 验证
            Rnet.eval()

        # batch_size循环
        for i, (concat_img, cover_img, secret_img) in enumerate(data_loader[phase]):

            if opt.use_gpu:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()
                concat_img = concat_img.cuda()

            concat_imgv = Variable(concat_img.clone())
            cover_imgv = Variable(cover_img.clone())
            secret_imgv = Variable(secret_img.clone())

            stego = Hnet(concat_imgv)

            secret_rev = Rnet(stego)

            optimizerH.zero_grad()
            optimizerR.zero_grad()

            errH = criterion(stego, cover_imgv)  # loss between cover and container
            errR = criterion(secret_rev, secret_imgv)  # loss between secret and revealed secret
            err_sum = errH + opt.beta * errR

            if phase == 'train':
                train_Hlosses.update(errH.detach().item(), opt.batch_size)
                train_Rlosses.update(errR.detach().item(), opt.batch_size)
                train_SumLosses.update(err_sum.detach().item(), opt.batch_size)

                err_sum.backward()
                optimizerH.step()
                optimizerR.step()

            if phase == 'valid':
                val_Hlosses.update(errH.detach().item(), opt.batch_size)
                val_Rlosses.update(errR.detach().item(), opt.batch_size)
                val_SumLosses.update(err_sum.detach().item(), opt.batch_size)

            # if i > 2:
            #     break

        # TODO: lr xiajiang
        # if phase == 'valid':
        #     schedulerH.step(val_SumLosses.avg)
        #     schedulerR.step(val_SumLosses.avg)
        # TODO: early stop
        # early_stopping(val_SumLosses.avg, Hnet)
        # early_stopping(val_SumLosses.avg, Rnet)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        # 打印日志，保存结果图，保存模型
        # print log
        if (epoch % opt.loss_freq == 0) | (epoch == opt.epochs - 1):
            epoch_time = time.time() - start_time
            if phase == 'train':
                epoch_log = 'train:' + '\n'
            else:
                epoch_log = 'valid:' + '\n'
            epoch_log += "epoch %d/%d : " % (epoch, opt.epochs - 1)
            epoch_log += "one epoch time is %.0fm %.0fs" % (epoch_time // 60, epoch_time % 60) + "\n"
            epoch_log += "learning rate: optimizerH_lr = %.8f\t optimizerR_lr = %.8f" % (
                optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
            if phase == 'train':
                epoch_log += "Hloss=%.6f\t Rloss=%.6f\t sumLoss=%.6f" % (train_Hlosses.avg, train_Rlosses.avg, train_SumLosses.avg) + "\n"
            if phase == 'valid':
                epoch_log += "Hloss=%.6f\t Rloss=%.6f\t sumLoss=%.6f" % (val_Hlosses.avg, val_Rlosses.avg, val_SumLosses.avg) + "\n"

            print_log(epoch_log, logPath)

        # save pictures and 差异图片
        if (epoch % opt.result_freq == 0) | (epoch == opt.epochs - 1):
            if phase == 'train':
                save_pic(phase, cover_img, stego, secret_img, secret_rev, opt.result_pics, opt.batch_size, epoch)
            if phase == 'valid':
                save_pic(phase, cover_img, stego, secret_img, secret_rev, opt.validation_pics, opt.batch_size, epoch)

        # save model params
        if phase == 'train' and ((epoch % opt.checkpoint_freq) == 0 | (epoch == opt.epochs - 1)):
            torch.save(
                Hnet.state_dict(),
                os.path.join(opt.checkpoint_path, 'H_epoch%04d_sumloss=%.6f.pth' % (epoch, train_SumLosses.avg))
            )
            torch.save(
                Rnet.state_dict(),
                os.path.join(opt.checkpoint_path, 'R_epoch%04d_sumloss=%.6f.pth' % (epoch, train_SumLosses.avg))
            )


def test(test_loader, Hnet, Rnet, criterion):
    print("---------- test begin ---------")
    Hnet.eval()
    Rnet.eval()
    for i, (concat_img, cover_img, secret_img) in enumerate(test_loader):
        if opt.use_gpu:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img)
        cover_imgv = Variable(cover_img)
        secret_imgv = Variable(secret_img)

        stego = Hnet(concat_imgv)

        secret_rev = Rnet(stego)

        errH = criterion(stego, cover_imgv)  # loss between cover and container
        errR = criterion(secret_rev, secret_imgv)  # loss between secret and revealed secret
        err_sum = errH + opt.beta * errR

    save_pic('test', cover_img, stego, secret_img, secret_rev, opt.test_pics, opt.batch_size, 0)
    log = 'test mode: loss is %.6f' % (err_sum.item()) + '\n'
    print(log)
    print("---------- test end ----------")


def main():
    # 定义全局参数
    global opt, logPath, optimizerH, optimizerR, schedulerH, schedulerR
    opt = parse_args()

    if torch.cuda.is_available() and opt.use_gpu:
        print("CUDA is available!")

    # torch.benchmark.cudnn.enabled = True

    # 如果不是debug模式, 需要创建文件夹去存储结果
    if not opt.debug:
        try:
            opt.checkpoint_path += "/checkPoints"
            opt.result_pics += "/resultPics"
            opt.validation_pics += "/validationPics"
            opt.log_path += "/trainingLogs"
            opt.test_pics += "/testPics"

            if not os.path.exists(opt.checkpoint_path):
                os.makedirs(opt.checkpoint_path)
            if not os.path.exists(opt.result_pics):
                os.makedirs(opt.result_pics)
            if not os.path.exists(opt.validation_pics):
                os.makedirs(opt.validation_pics)
            if not os.path.exists(opt.log_path):
                os.makedirs(opt.log_path)
            if not os.path.exists(opt.test_pics):
                os.makedirs(opt.test_pics)
        except OSError:
            print("mkdir failed!")

    logPath = opt.log_path + '/%s_%d_log.txt' % (opt.dataset, opt.batch_size)

    print_log(time.asctime(time.localtime(time.time())), logPath, False)
    print_log(str(opt), logPath, False)  # 把所有参数打印在日志中

    # 准备数据集，分别准备train/val/test
    # 通过判断test是否有值，区分是train/val 还是test模式
    if opt.test == '':
        data_dir = opt.images_path
        image_datasets = {
            x: DirectoryDataset(os.path.join(data_dir, x), preprocess=transforms) for x in ['train', 'valid']
        }

        dataloaders = {x: DataLoader(
                        image_datasets[x],
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        shuffle=True,
                        drop_last=True) for x in ['train', 'valid']}

        assert dataloaders
    else:  # 测试模式
        # 先加载训练好的模型
        # TODO: trained model path
        opt.Hnet = ""
        opt.Rnet = ''
        # opt.Rnet = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"

        # 读取test数据
        test_dir = opt.test
        test_dataset = DirectoryDataset(test_dir)
        test_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=True,
            drop_last=True,
        )
        assert test_loader

    # 初始化模型 Hnet Rnet
    modelH = HNet()
    modelR = RNet()

    if opt.use_gpu:
        modelH.cuda()
        modelR.cuda()
        torch.backends.cudnn.benchmark = True

    # whether to load pre-trained model
    if opt.Hnet != '':
        modelH.load_state_dict(torch.load(opt.Hnet))
        print_network(modelH)
    if opt.Rnet != '':
        modelR.load_state_dict(torch.load(opt.Rnet))
        print_network(modelR)

    # MSE loss
    criterion = nn.MSELoss().cuda()

    # 开始训练！
    if opt.test == '':
        # setup optimizer
        optimizerH = torch.optim.Adam(modelH.parameters(), lr=7e-5)
        optimizerR = torch.optim.Adam(modelR.parameters(), lr=7e-5)

        # 训练策略，学习率下降, 若训练集的loss值一直不变，就调小学习率
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)
        schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)
        print_log("training is beginning ......................................", logPath)

        for epoch in range(opt.epochs):
            # 只有训练的时候才会计算和更新梯度
            train(dataloaders, epoch, Hnet=modelH, Rnet=modelR, criterion=criterion)

    # 开始测试！
    else:
        # 加载保存的模型
        H_path = ''
        R_path = ''
        modelH = HNet()
        modelR = RNet()
        modelH.load_state_dict(torch.load(H_path))
        modelR.load_state_dict(torch.load(R_path))
        test(test_loader, 0, Hnet=modelH, Rnet=modelR, criterion=criterion)
        print("test is completed, the result pic is saved in the ./training/testPics/")


if __name__ == '__main__':
    main()

