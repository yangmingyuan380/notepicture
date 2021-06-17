import torch
import torch.nn as nn
from torch.autograd import Variable
from models import CNN, GoogleNet
from datasets import CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision import utils as vutils
import time
import os
from config import logger, batch_size, base_lr, max_epoch, model_path, restor, device, onnx_path
import shutil

# torch.cuda.empty_cache()
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
count = 1


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
        将tensor保存为图片
        :param input_tensor: 要保存的tensor
        :param filename: 保存的文件名
        """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize (input_tensor)
    vutils.save_image(input_tensor, filename)


def calculat_acc(output, target, epoch=0, img=None):
    output, target = output.view(-1, 36), target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)  # 按行归一化
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for k, (i, j) in enumerate(zip(target, output)):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            if epoch > 0:
                global count
                filename = f'data/error/{count}.png'
                count = count + 1
                save_image_tensor(img[k], filename)
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


def train():
    transforms = Compose([ToTensor()])
    # train_path = './data/train'
    train_path = ['data/train_after', 'data/train']
    train_dataset = CaptchaData(train_path, transform=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                   shuffle=True, drop_last=True)
    # test_path = './data/test'
    test_path = ['data/test_after']  # ['data/test_after','data/test']
    test_data = CaptchaData(test_path, transform=transforms)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,
                                  num_workers=0, shuffle=True, drop_last=True)
    model = GoogleNet()
    # model = CNN()
    model.to(device)
    # 可视化模型结构
    images, labels = next(iter(train_data_loader))
    images = Variable(images)
    images = images.to(device)
    with logger:
        logger.add_graph(model, images)
    torch.onnx.export(model, images, onnx_path)
    # 是否加载已有模型参数
    if restor:
        model.load_state_dict(torch.load(model_path))
    #        freezing_layers = list(cnn.named_parameters())[:10]
    #        for param in freezing_layers:
    #            param[1].requires_grad = False
    #            print('freezing layer:', param[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.MultiLabelSoftMarginLoss()
    best_acc = 0.8
    for epoch in range(max_epoch):
        start_ = time.time()

        loss_history = []
        acc_history = []
        model.train()
        for img, target in train_data_loader:
            img = Variable(img)
            target = Variable(target)
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calculat_acc(output, target)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('train_loss: {:.4}|train_acc: {:.4}'.format(
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))
        logger.add_scalar('train_loss', torch.mean(torch.Tensor(loss_history)), epoch)
        logger.add_scalar('train_acc', torch.mean(torch.Tensor(acc_history)), epoch)

        loss_history = []
        acc_history = []
        model.eval()
        for img, target in test_data_loader:
            img = Variable(img)
            target = Variable(target)
            img, target = img.to(device), target.to(device)
            output = model(img)
            acc = calculat_acc(output, target)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        print('test_loss: {:.4}|test_acc: {:.4}'.format(
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))
        # 保存测试集上效果最好的模型
        if best_acc < torch.mean(torch.Tensor(acc_history)):
            best_acc = torch.mean(torch.Tensor(acc_history))
            torch.save(model.state_dict(), model_path)
            shutil.rmtree('data/error')
            os.mkdir('data/error')
            calculat_acc(output, target, epoch, img)
        logger.add_scalar('test_loss', torch.mean(torch.Tensor(loss_history)), epoch)
        logger.add_scalar('test_acc', torch.mean(torch.Tensor(acc_history)), epoch)
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time() - start_))


if __name__ == "__main__":
    train()
    pass
