import argparse
import torch
import torchvision
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Resize
from models import resnet18
from datetime import datetime

######数据集变换
def input_transform():
    return Compose([
        Resize(224),  # 改变尺寸
        ToTensor(),  # 变成tensor
    ])
def train_test_net(gpu, args):
    ##随机种子
    torch.manual_seed(0)
    #####构造数据集
    writer = SummaryWriter()
    train_dataset = torchvision.datasets.MNIST(root='./dataset/',
                                               train=True,
                                               transform=input_transform(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8)
    print("train_loader end")
    test_data = torchvision.datasets.MNIST(root='./dataset/',
                                           train=False,
                                           transform=input_transform(),
                                           download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=2)
    print("test_loader end")
    #########构造网络，并且放到GPU上训练上
    net = resnet18(use_cbam=True, use_mixpool=True)
    torch.cuda.set_device(gpu)
    net.cuda(gpu)
    #####损失函数和优化算法
    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(net.parameters(),args.lr)
    #####训练开始计时
    start = datetime.now()
    #####每个Epoch的步长
    total_step = len(train_loader)
    #######开始训练
    print("Training start")
    for epoch in range(args.epochs):
        #####每100 step的总loss
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = net(images)
            _, predict = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##########统计每100step的总loss
            running_loss += loss.item()
            ####################第1次迭代，保存计算图
            if i < 1:
                writer.add_graph(net, (images,))
                print("Saving Model start")
                torch.save(net, ".\models\Res18-Cbam-MixPool-test.pth")
                print("Saving Model end")
            ##########每(total_step//100)次迭代，打印一次信息
            if (i +1) % (total_step//100) == 0 and gpu == 0:
                print('Epoch [{}/{}],\tStep [{}/{}],\tLoss: {:.4f},\tTrain_Acc:{:.4f},\tTime:{}'.format(
                    epoch + 1,#当前epoch
                    args.epochs,#总的epoch
                    i + 1,#当前step
                    total_step,#当前epoch内总的step
                    running_loss/(total_step//100),#100个step的平均loss
                    (predict == labels).sum().item() / labels.size(0),
                    str(datetime.now() - start)
                ))
                writer.add_scalar('Train_Loss', running_loss/(total_step//100), epoch * len(train_loader) + i)
                writer.add_scalar('Train_Acc', (predict == labels).sum().item() / labels.size(0), epoch * len(train_loader) + i)
                #####每(total_step//100) step 清空一次
                running_loss = 0.0


            ########每个Epoch测试两次，打印信息，并且保存一下模型
            if (i + 1) % (total_step//2) == 0 and gpu == 0:
                correct = 0
                total = 0
                count = 0
                with torch.no_grad():
                    for j, (images_test, labels_test) in enumerate(test_loader):
                        images_test = images_test.cuda()
                        labels_test = labels_test.cuda()
                        out = net(images_test)
                        _, pred = torch.max(out, 1)
                        ########记录每一次的测试精度
                        writer.add_scalar('Val_Acc', (pred == labels_test).sum().item()/labels_test.size(0), epoch * len(train_loader) + j)
                        correct += (pred == labels_test).sum().item()
                        total += labels_test.size(0)
                        print('\t batch:{}, Time:{} Val_Acc:{:.4f}\n'.format(count + 1,
                                                                             str(datetime.now() - start),
                                                                             (pred == labels_test).sum().item()/labels_test.size(0)))
                        count += 1
                ######计算整体测试集上的平均准确率
                accuracy = float(correct) / total
                print('Val_Acc = {:.4f}, Time:{}\n'.format(accuracy,str(datetime.now() - start)))
                writer.add_scalar('Mean_Val_Acc', accuracy, epoch * len(train_loader) + i)
                print("Saving Model start")
                torch.save(net, ".\models\Res18-Cbam-MixPool-{}.pth".format(epoch * len(train_loader) + i))
                print("Saving Model end")
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    print("Saving Model start")
    torch.save(net, ".\models\Res18-Cbam-MixPool-final.pth")
    print("Saving Model end")
    writer.close()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=4, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch_size')
    parser.add_argument('--lr', default=1e-2, type=float, metavar='N',
                        help='learning_rate')
    args = parser.parse_args()
    train_test_net(0, args)

if __name__ == '__main__':
    main()

