# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: main-old.py
@time: 2018/3/20

"""

import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils.transformed as transforms
from data.utils import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet

DATA_DIR = '/n/liyz/data/deep-steganography-dataset/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


# custom weights initialization called on netG and netD
# 在netG和netD上调用自定义权重初始化函数   权值初始化
# 之后通过net实例.apply(fn) 来进行模型参数的初始化，fn就是初始化函数，比如这个weights_init函数
# netG.apply(weights_init) 递归调用函数，遍历netG中的所有submodule来做参数
# m.weight.data 是卷积核参数
# m.bias.data 是偏置项参数
def weights_init(m):
    classname = m.__class__.__name__ # 获得nn.module的名字
    print(classname)# 会打印出每次传入的子模型名
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)# 正态分布，均值为0，方差为0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)# BN层初始化y,
        m.bias.data.fill_(0)# BN层初始化β，默认为0


# print the structure and parameters number of the net 打印网络结构中参数的个数
def print_network(net):
    num_params = 0# 参数数量
    for param in net.parameters():
        num_params += param.numel()# param.numel()返回param中元素的数量
    print_log(str(net), logPath)# 打印日志
    print_log('Total number of parameters: %d' % num_params, logPath)


# save code of current experiment 保存当前实验的代码
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/main-old.py
    # os.path.realpath(__file__) 获得当前执行脚本的绝对路径
    cur_work_dir, mainfile = os.path.split(main_file_path)  # eg：/n/liyz/videosteganography/

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)# shutil.copyfile 根据路径复制文件

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path) # shutil.copytree 递归复制，将文件夹下所有的文件都复制

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, writer, schedulerH, schedulerH, schedulerR, val_loader, smallestLoss

    #################  output configuration   ###############
    opt = parser.parse_args() # 是参数空间namespace中所有的参数

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create dirs to save the result #############
    if not opt.debug:# 如果不是debug模式
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs"
            opt.outcodes += experiment_dir + "/codes"
            opt.testPics += experiment_dir + "/testPics"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)

    if opt.test == '':
        # tensorboardX writer
        writer = DATA_DIR(comment='**' + opt.remark)
        ##############   get dataset   ############################
        traindir = os.path.join(DATA_DIR, 'train')
        valdir = os.path.join(DATA_DIR, 'val')
        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),  # resize to a given size
                transforms.ToTensor(),
            ]))
        val_dataset = MyImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert train_dataset
        assert val_dataset
    else:
        opt.Hnet = "./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth"
        opt.Rnet = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"
        testdir = opt.test
        test_dataset = MyImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert test_dataset

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)
    # whether to load pre-trained model
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(Hnet)

    Rnet = RevealNet(output_function=nn.Sigmoid)
    Rnet.cuda()
    Rnet.apply(weights_init)
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()
    print_network(Rnet)

    # MSE loss
    criterion = nn.MSELoss().cuda()
    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                                  shuffle=True, num_workers=int(opt.workers))
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                                shuffle=False, num_workers=int(opt.workers))
        smallestLoss = 10000
        print_log("training is beginning .......................................................", logPath)
        for epoch in range(opt.niter):
            ######################## train ##########################################
            train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ####################### validation  #####################################
            val_hloss, val_rloss, val_sumloss = validation(val_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ####################### adjust learning rate ############################
            schedulerH.step(val_sumloss)
            schedulerR.step(val_rloss)

            # save the best model parameters
            if val_sumloss < globals()["smallestLoss"]:
                globals()["smallestLoss"] = val_sumloss
                # do checkPointing
                torch.save(Hnet.state_dict(),
                           '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_hloss))
                torch.save(Rnet.state_dict(),
                           '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_rloss))

        writer.close()
    # test mode
    else:
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=int(opt.workers))
        test(test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
        print("##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


def train(train_loader, epoch, Hnet, Rnet, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  # record loss of H-net
    Rlosses = AverageMeter()  # record loss of R-net
    SumLosses = AverageMeter()  # record Hloss + β*Rloss

    # switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()

        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step 

        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # concat cover images and secret images as input of H-net
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img)
        cover_imgv = Variable(cover_img)

        container_img = Hnet(concat_imgv)  # put concat_image into H-net and get container image
        errH = criterion(container_img, cover_imgv)  # loss between cover and container
        Hlosses.update(errH.data[0], this_batch_size)

        rev_secret_img = Rnet(container_img)  # put concatenated image into R-net and get revealed secret image
        secret_imgv = Variable(secret_img)
        errR = criterion(rev_secret_img, secret_imgv)  # loss between secret image and revealed secret image
        Rlosses.update(errR.data[0], this_batch_size)

        betaerrR_secret = opt.beta * errR
        err_sum = errH + betaerrR_secret
        SumLosses.update(err_sum.data[0], this_batch_size)

        err_sum.backward()
        optimizerH.step()
        optimizerR.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.4f Loss_R: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            Hlosses.val, Rlosses.val, SumLosses.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        # genereate a picture every resultPicFrequency steps
        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                            opt.trainpics)

    # epcoh log
    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Hloss=%.6f\tepoch_Rloss=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses.avg, Rlosses.avg, SumLosses.avg)
    print_log(epoch_log, logPath)


    if not opt.debug:
        # record lr
        writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        # record loss
        writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)


def validation(val_loader, epoch, Hnet, Rnet, criterion):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    for i, data in enumerate(val_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data
        this_batch_size = int(all_pics.size()[0] / 2)

        cover_img = all_pics[0:this_batch_size, :, :, :]
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        concat_img = torch.cat([cover_img, secret_img], dim=1)

        # 数据放入GPU
        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, volatile=True)
        cover_imgv = Variable(cover_img, volatile=True)

        container_img = Hnet(concat_imgv)
        errH = criterion(container_img, cover_imgv)
        Hlosses.update(errH.data[0], this_batch_size)

        rev_secret_img = Rnet(container_img)
        secret_imgv = Variable(secret_img, volatile=True)
        errR = criterion(rev_secret_img, secret_imgv)
        Rlosses.update(errR.data[0], this_batch_size)

        if i % 50 == 0:
            save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                            opt.validationpics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print(
        "#################################################### validation end ########################################################")
    return val_hloss, val_rloss, val_sumloss


def test(test_loader, epoch, Hnet, Rnet, criterion): # criterion 标准
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval() # 测试模型时，会在前面使用这个语句，训练时会使用net.train() 是因为在网络训练和测试时采用不同方式，比如BN 和Droupout时需这样
    Rnet.eval()
    Hlosses = AverageMeter()  # record the Hloss in one epoch 计算loss
    Rlosses = AverageMeter()  # record the Rloss in one epoch
    for i, data in enumerate(test_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step 

        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchSize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # concat cover and original secret to get the concat_img with 6 channels
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, volatile=True)  # concat_img as input of Hiding net
        cover_imgv = Variable(cover_img, volatile=True)  # cover_imgv as label of Hiding net

        container_img = Hnet(concat_imgv)  # take concat_img as input of H-net and get the container_img
        errH = criterion(container_img, cover_imgv)  # H-net reconstructed error
        Hlosses.update(errH.data[0], this_batch_size)

        rev_secret_img = Rnet(container_img)  # containerImg as input of R-net and get "rev_secret_img"
        secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
        errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error
        Rlosses.update(errR.data[0], this_batch_size)
        save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                        opt.testPics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print(
        "#################################################### test end ########################################################")
    return val_hloss, val_rloss, val_sumloss


# print training log and save into logFiles 打印日志，接收两个参数，1是日志信息，2是地址
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # debug mode will not write logs into files
    if not opt.debug:  # opt所有参数的集合，如果没有debug这个参数的话，就输出到日志文件中中
        # write logs into log file
        if not os.path.exists(log_path): # 如果这个地址没有文件，创建一个文件，并在里面写入日志信息
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else: # 如果地址下有文件，打开这个文件后，再写入日志
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pics, coverImg filePath and secretImg filePath
def save_result_pic(this_batch_size, originalLabelv, ContainerImg, secretLabelv, RevSecImg, epoch, i, save_path):
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames = RevSecImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        showContainer = torch.cat([originalFrames, containerFrames], 0)
        showReveal = torch.cat([secretFrames, revSecFrames], 0)
        # resultImg contains four rows: coverImg, containerImg, secretImg, RevSecImg, total this_batch_size columns
        resultImg = torch.cat([showContainer, showReveal], 0)
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
