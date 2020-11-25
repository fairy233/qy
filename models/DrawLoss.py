import matplotlib.pyplot as plt

# 画出训练的loss曲线和lr曲线
# TODO: LOSS

#     plt.figure(1)
#     plt.plot(epoch, optimizerH.param_groups[0]['lr'], color='r', label='lr/H_lr')
#     plt.plot(epoch, optimizerR.param_groups[0]['lr'], color='g', label='lr/R_lr')
#     plt.xlabel('epochs')  # x轴表示
#     plt.ylabel('LR')  # y轴表示
#     plt.title("Training LR change")  # 图标标题表示
#     plt.legend()  # 每条折线的label显示
#     plt.show()
#
#     plt.figure(2)
#     plt.plot(epoch, train_Rlosses.avg, color='b', label='train/R_loss')
#     plt.plot(epoch, train_Hlosses.avg, color='g', label='train/H_loss')
#     plt.plot(epoch, train_SumLosses.avg, color='r', label='train/sum_loss')
#     plt.xlabel('epochs')  # x轴表示
#     plt.ylabel('Loss')  # y轴表示
#     plt.title("Training Loss change")  # 图标标题表示
#     plt.legend()  # 每条折线的label显示
#     plt.show()
#
# # 画出验证的loss曲线
# if phase == 'valid' and not opt.debug:
#     plt.plot(epoch, val_Rlosses.avg, color='b', label='val/R_loss')
#     plt.plot(epoch, val_Hlosses.avg, color='g', label='val/H_loss')
#     plt.plot(epoch, val_SumLosses.avg, color='r', label='val/sum_loss')
#     plt.xlabel('epochs')
#     plt.ylabel('Loss')
#     plt.title("Validation Loss change")
#     plt.legend()
#     plt.show()