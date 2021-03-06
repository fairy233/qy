import argparse
import os
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from loss import (loss)
from models.AverageMeter import AverageMeter
from models.Hnet_unet2 import HNet
from models.RNet import RNet
from models.util import (
    DirectoryDataset,
    cv2torch,
    random_noise,
    random_crop2,
    torch2cv,
    hdr2ldr
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Batch size.'
    )
    parser.add_argument(
        '--image_size', type=int, default=64, help='image size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of data loading workers.',
    )
    parser.add_argument(
        '--images_path', type=str, default='./ldr/', help='Path to coverImage data.'
    )
    parser.add_argument(
        '--use_gpu', type=bool, default=True, help='Use GPU for training.'
    )
    parser.add_argument(
        '--debug', type=bool, default=False, help='debug mode do not create folders'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate, default=0.001'
    )
    parser.add_argument(
        '--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5'
    )
    parser.add_argument(
        '--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument(
        # 路径可为'./hdr/test'
        '--test', default='./ldr/test', help='test mode, you need give the test pics dirs in this param'
    )
    parser.add_argument(
        '--beta', type=float, default=0.75, help='hyper parameter of loss_sum'
    )
    parser.add_argument(
        '--epochs', type=int, default=600, help='the num of training times'
    )
    parser.add_argument(
        '--checkpoint_freq', type=int, default=300, help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '--loss_freq', type=int, default=20, help='Report (average) loss every x epochs.',
    )
    parser.add_argument(
        '--result_freq', type=int, default=20, help='save the resultPictures every x epochs'
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
        '--test_log', default='./test/testLog.txt', help='test log'
    )
    parser.add_argument(
        # '--Hnet', default='./training/checkPoints/H_epoch0400_sumloss0.000995_lr0.001000.pth', help="path to Hidenet (to continue training)"
        '--Hnet', default='', help="path to Hidenet (to continue training)"

    )
    parser.add_argument(
        # '--Rnet', default='./training/checkPoints/R_epoch0400_sumloss0.000995_lr0.001000.pth', help="path to Revealnet (to continue training)"
        '--Rnet', default='', help="path to Revealnet (to continue training)"
    )

    return parser.parse_args()


# 对输入图像的预处理的函数--图像增强（图像翻转flip, 高斯噪声）
def transforms(ldr):
    ldr_size = np.array(ldr.shape)
    if sum(ldr_size < 64) >= 2:
        raise Exception('img size is too small!')
    ldr = random_crop2(ldr, resize=True)
    ldr = ldr / 256
    if np.random.rand() < 0.5:
        ldr = cv2.flip(ldr, 1)  # 1 水平翻转 0 垂直翻转 -1 水平垂直翻转
    if np.random.rand() < 0.5:
        ldr = random_noise(ldr)

    ldr = cv2torch(ldr)  # 转为(3,256,256) tensor
    return ldr


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

        #  np.vstack  np.hstack
        showContainer = np.hstack((cover, stego))
        showReveal = np.hstack((secret, secret_rev))
        resultImg = np.vstack((showContainer, showReveal))

        # tone map  ldr
        cover_ldr = (hdr2ldr(cover) * 255).astype(int)
        secret_ldr = (hdr2ldr(secret) * 255).astype(int)
        stego_ldr = (hdr2ldr(stego) * 255).astype(int)
        secret_rev_ldr = (hdr2ldr(secret_rev) * 255).astype(int)
        
        showContainer_ldr = np.hstack((cover_ldr, stego_ldr))
        showReveal_ldr = np.hstack((secret_ldr, secret_rev_ldr))
        resultImg_ldr = np.vstack((showContainer_ldr, showReveal_ldr))
        
        cover_diff = cover - stego
        secret_diff = secret - secret_rev
        
        # diff图像，只在test中保存，且只保存hdr格式
        # cover_diff_ldr = (hdr2ldr(cover_diff) * 255).astype(int)
        # secret_diff_ldr = (hdr2ldr(secret_diff) * 255).astype(int)
        # diffImg_ldr = np.vstack((cover_diff_ldr, secret_diff_ldr))

        diffImg = np.hstack((cover_diff, secret_diff))
        if phase == 'test':
            cv2.imwrite('%s/cover_%02d.hdr' % (save_path, epoch), cover)
            cv2.imwrite('%s/stego_%02d.hdr' % (save_path, epoch), stego)
            cv2.imwrite('%s/secret_%02d.hdr' % (save_path, epoch), secret)
            cv2.imwrite('%s/secret_rev_%02d.hdr' % (save_path, epoch), secret_rev)
            cv2.imwrite('%s/test_diff_%02d.hdr' % (save_path, epoch), diffImg)
        else:
            if phase == 'train':
                resultImgName = '%s/trainResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)
            if phase == 'valid':
                resultImgName = '%s/valResult_epoch%04d_batch%02d.hdr' % (save_path, epoch, batch_size)


            # result hdr
            cv2.imwrite(resultImgName, resultImg)
            # result ldr
            cv2.imwrite(resultImgName + '.jpg', resultImg_ldr)


def save_pic2(phase, cover, stego, secret, secret_rev, save_path, batch_size, epoch):
    # save result pictures for ldr images
    if not opt.debug:
        # tensor  --> numpy.narray
        cover = torch2cv(cover*255)
        secret = torch2cv(secret*255)
        stego = torch2cv(stego*255)
        secret_rev = torch2cv(secret_rev*255)

        #  np.vstack  np.hstack
        showContainer = np.hstack((cover, stego))
        showReveal = np.hstack((secret, secret_rev))
        resultImg = np.vstack((showContainer, showReveal))

        cover_diff = cover - stego
        secret_diff = secret - secret_rev

        diffImg = np.hstack((cover_diff, secret_diff))
        if phase == 'test':
            cv2.imwrite('%s/cover_%02d.jpg' % (save_path, epoch), cover)
            cv2.imwrite('%s/stego_%02d.jpg' % (save_path, epoch), stego)
            cv2.imwrite('%s/secret_%02d.jpg' % (save_path, epoch), secret)
            cv2.imwrite('%s/secret_rev_%02d.jpg' % (save_path, epoch), secret_rev)
            cv2.imwrite('%s/test_diff_%02d.jpg' % (save_path, epoch), diffImg)
        else:
            if phase == 'train':
                resultImgName = '%s/trainResult_epoch%04d_batch%02d.jpg' % (save_path, epoch, batch_size)
            if phase == 'valid':
                resultImgName = '%s/valResult_epoch%04d_batch%02d.jpg' % (save_path, epoch, batch_size)

            cv2.imwrite(resultImgName, resultImg)


def train(data_loader, epoch, Hnet, Rnet):
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
        for i, data in enumerate(data_loader[phase]):

            all_pics = data  # allpics contains cover images and secret images
            this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step

            # first half of images will become cover images, the rest are treated as secret images
            cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
            secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

            # concat cover images and secret images as input of H-net
            concat_img = torch.cat([cover_img, secret_img], dim=1)

            if opt.use_gpu:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()
                concat_img = concat_img.cuda()

            concat_imgv = Variable(concat_img.clone(), requires_grad=False)
            cover_imgv = Variable(cover_img.clone(), requires_grad=False)
            secret_imgv = Variable(secret_img.clone(), requires_grad=False)

            if phase == 'train':
                stego = Hnet(concat_imgv)

                secret_rev = Rnet(stego)

                optimizerH.zero_grad()
                optimizerR.zero_grad()

                errH = loss(stego, cover_imgv)  # loss between cover and container
                errR = loss(secret_rev, secret_imgv)  # loss between secret and revealed secret
                err_sum = errH + opt.beta * errR
            else:
                with torch.no_grad():
                    stego = Hnet(concat_imgv)

                    secret_rev = Rnet(stego)

                    errH = loss(stego, cover_imgv)  # loss between cover and container
                    errR = loss(secret_rev, secret_imgv)  # loss between secret and revealed secret
                    err_sum = errH + opt.beta * errR

            if phase == 'train':
                train_Hlosses.update(errH.detach().item(), opt.batch_size)
                train_Rlosses.update(errR.detach().item(), opt.batch_size)
                train_SumLosses.update(err_sum.detach().item(), opt.batch_size)

                err_sum.backward()
                optimizerH.step()
                optimizerR.step()

            if phase == 'valid':
                with torch.no_grad():
                    val_Hlosses.update(errH.detach().item(), opt.batch_size)
                    val_Rlosses.update(errR.detach().item(), opt.batch_size)
                    val_SumLosses.update(err_sum.detach().item(), opt.batch_size)

            # if i > 2:
            #     break
        if phase == 'train' and epoch % 100 == 0 and epoch != 0:
            schedulerH.step()
            schedulerR.step()

        # TODO: early stop
        # early_stopping(val_SumLosses.avg, Hnet)
        # early_stopping(val_SumLosses.avg, Rnet)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        # save pictures and 差异图片
        if (epoch % opt.result_freq == 0) or (epoch == opt.epochs - 1):
            if phase == 'train':
                save_pic2(phase, cover_img, stego, secret_img, secret_rev, opt.result_pics, opt.batch_size, epoch)
            if phase == 'valid':
                save_pic2(phase, cover_img, stego, secret_img, secret_rev, opt.validation_pics, opt.batch_size, epoch)

        # save model params
        if phase == 'train':
            if epoch % opt.checkpoint_freq == 0 or epoch == opt.epochs - 1:
                torch.save(
                    Hnet.state_dict(),
                    os.path.join(opt.checkpoint_path, 'H_epoch%04d_sumloss%.6f_lr%.6f.pth' % (epoch, train_SumLosses.avg, opt.lr))
                )
                torch.save(
                    Rnet.state_dict(),
                    os.path.join(opt.checkpoint_path, 'R_epoch%04d_sumloss%.6f_lr%.6f.pth' % (epoch, train_SumLosses.avg, opt.lr))
                )

        # print log
        if (epoch % opt.loss_freq == 0) or (epoch == opt.epochs - 1):
            epoch_time = time.time() - start_time
            if phase == 'train':
                epoch_log = 'train:' + '\n'
            else:
                epoch_log = 'valid:' + '\n'
            epoch_log += "epoch %d/%d : " % (epoch, opt.epochs)
            epoch_log += "one epoch time is %.0fm %.0fs" % (epoch_time // 60, epoch_time % 60) + "\n"
            epoch_log += "learning rate: optimizerH_lr = %.8f\t optimizerR_lr = %.8f" % (
                optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
            # schedulerH.get_lr()[0] schedulerR.get_lr()[0]
            if phase == 'train':
                epoch_log += "Hloss=%.6f\t Rloss=%.6f\t sumLoss=%.6f" % (train_Hlosses.avg, train_Rlosses.avg, train_SumLosses.avg) + "\n"
            if phase == 'valid':
                epoch_log += "Hloss=%.6f\t Rloss=%.6f\t sumLoss=%.6f" % (val_Hlosses.avg, val_Rlosses.avg, val_SumLosses.avg) + "\n"

            print_log(epoch_log, logPath)


def test(data_loader, Hnet, Rnet):
    print_log("---------- test begin ---------", opt.test_log)
    print_log(time.asctime(time.localtime(time.time())), opt.test_log, False)
    Hnet.eval()
    Rnet.eval()
    for i, data in enumerate(data_loader):
        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step

        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # concat cover images and secret images as input of H-net
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        if opt.use_gpu:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, requires_grad=False)
        cover_imgv = Variable(cover_img, requires_grad=False)
        secret_imgv = Variable(secret_img, requires_grad=False)

        with torch.no_grad():
            stego = Hnet(concat_imgv)
            secret_rev = Rnet(stego)

            errH = loss(stego, cover_imgv)  # loss between cover and container
            errR = loss(secret_rev, secret_imgv)  # loss between secret and revealed secret
            err_sum = errH + opt.beta * errR
        
        save_pic2('test', cover_img, stego, secret_img, secret_rev, opt.test_pics, opt.batch_size, i)

    log = 'test: loss is %.6f' % (err_sum.item()) + '\n'
    print_log(log, opt.test_log)
    print_log("---------- test end ----------", opt.test_log)


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

    logPath = opt.log_path + '/train_%d_log.txt' % opt.batch_size

    print_log(time.asctime(time.localtime(time.time())), logPath, False)
    print_log(str(opt), logPath, False)  # 把所有参数打印在日志中

    # 准备数据集，分别准备train/val/test
    # 通过判断test是否有值，区分是train/val 还是test模式
    if opt.test == '':
        print_log('prepare train and val dataset', logPath)
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
        # 读取test数据
        print_log('prepare test dataset', opt.test_log)
        test_dir = opt.test
        test_dataset = DirectoryDataset(test_dir, preprocess=transforms)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
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

    # 开始训练！
    if opt.test == '':
        # setup optimizer  beta1=0.9, beta2=0.999
        optimizerH = torch.optim.Adam(modelH.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        optimizerR = torch.optim.Adam(modelR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

        # 训练策略，学习率下降, 若训练集的loss值一直不变，就调小学习率
        # schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)
        # schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)
        schedulerH = torch.optim.lr_scheduler.ExponentialLR(optimizerH, gamma=0.8)
        schedulerR = torch.optim.lr_scheduler.ExponentialLR(optimizerR, gamma=0.8)
        print_log("training is beginning ......................................", logPath)

        for epoch in range(opt.epochs):
            # 只有训练的时候才会计算和更新梯度
            train(dataloaders, epoch, Hnet=modelH, Rnet=modelR)
            # print_log("train is completed, the result is saved in the ./training", logPath)

    # 开始测试！
    else:
        test(test_loader, Hnet=modelH, Rnet=modelR)


if __name__ == '__main__':
    main()

