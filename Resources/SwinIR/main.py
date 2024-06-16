import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import utils
from loss import SSIM
from net import SwinIR
from data import *


def main():

    random.seed(6)
    torch.manual_seed(6)
    # torch.cuda.manual_seed_all(6)
    EPOCH = 100
    BATCH_SIZE = 18
    PATCH_SIZE = 128
    LEARNING_RATE = 2e-3
    lr_list = []
    loss_list = []

    inputPathTrain = './inputTrain/'
    targetPathTrain = './targetTrain/'
    inputPathTest = './inputTest/'
    targetPathTest = './targetTest/'
    resultPathTest = './resultTest/'  # 测试结果图片路径
    best_psnr = 0
    best_epoch = 0
	
	
	psnr = utils.PSNR()  # 实例化峰值信噪比计算类
	psnr = psnr.cuda()

    # 关于网络中具体参数的设置详见源链接中 main_test_swinir.py
    myNet = SwinIR(upscale=1, in_chans=3, img_size=PATCH_SIZE, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
    myNet = myNet.cuda()
    # 多卡
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        myNet = nn.DataParallel(myNet, device_ids=device_ids)

    criterion1 = SSIM().cuda()  # 结构相似性
    criterion2 = nn.MSELoss().cuda()  # 均方误差

    optimizer = optim.AdamW(myNet.parameters(), lr=LEARNING_RATE)  # 优化器
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.1)  # 学习率调整

    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain, patch_size=PATCH_SIZE)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=6,
                             pin_memory=True)

    datasetValue = MyValueDataSet(inputPathTest, targetPathTest, patch_size=PATCH_SIZE)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=16, shuffle=True, drop_last=True, num_workers=6,
                             pin_memory=True)

    # 测试数据
    ImageNames = os.listdir(inputPathTest)  # 测试路径文件名
    datasetTest = MyTestDataSet(inputPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=6,
                            pin_memory=True)

    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists('./model_best.pth'):
        myNet.load_state_dict(torch.load('./model_best.pth'))

    for epoch in range(EPOCH):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            input_train, target = Variable(x).cuda(), Variable(y).cuda()

            output_train = myNet(input_train)

            l_ssim = criterion1(output_train, target)
            l_2 = criterion2(output_train, target)

            loss = (1 - l_ssim) + l_2

            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, EPOCH, loss.item()))

        myNet.eval()
        psnr_val_rgb = []
        for index, (x, y) in enumerate(valueLoader, 0):
            input_, target_value = x.cuda(), y.cuda()
            with torch.no_grad():
                output_value = myNet(input_)
            for output_value, target_value in zip(output_value, target_value):
                psnr_val_rgb.append(psnr(output_value, target_value))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(myNet.state_dict(), 'model_best.pth')

        loss_list.append(epochLoss)
        lr_list.append(scheduler.get_last_lr())
        scheduler.step()
        torch.save(myNet.state_dict(), 'model.pth')
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch+1, timeEnd-timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

    print('--------------------------------------------------------------')
    myNet.load_state_dict(torch.load('./model_best.pth'))
    myNet.eval()

    with torch.no_grad():
        timeStart = time.time()
        for index, x in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()
            input_test = x.cuda()
            output_test = myNet(input_test)
            save_image(output_test, resultPathTest + str(ImageNames[index]))
        timeEnd = time.time()
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))

    plt.figure(1)
    x = range(0, EPOCH)
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.plot(x, loss_list, 'r-')
    plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.plot(x, lr_list, 'r-')

    plt.show()

if __name__ == '__main__':
    main()





