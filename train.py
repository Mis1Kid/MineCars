from copy import deepcopy

from config.config import *
from dataset.dataset import MineCars
from model.resnet import ResNet
from tools.utils import validate
from torch.utils.tensorboard import SummaryWriter
from model.denseNet import DenseNet121
import os
# res = ResNet(**config['Res']).to(device)
res = ResNet([3, 4, 23, 3])
dense=DenseNet121()
tbWriter = SummaryWriter()
# vit = Vit(**config['Vit']).to(device)


def trainOneEpoch(model, trainloader, epoch, optimizer, device, checkpoint=None):
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    total_num = 0
    for batchIndex, (data, labels) in enumerate(trainloader):
        data = data.to(device)
        labels = labels.to(device)
        total_num += labels.shape[0]
        output = model(data)
        pred_classes = torch.argmax(output, dim=-1)
        # correct num
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = lossFn(output, labels)

        loss.backward()
        # train loss
        accu_loss += loss.detach()
        # print train info
        if batchIndex % 10==0 and batchIndex!=0:
            print("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                  accu_loss.item() / (batchIndex + 1),
                                                                  accu_num.item() / total_num))
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (batchIndex + 1), accu_num.item() / total_num


def train(trainloader, testloader, model, optimizer, scheduler, device, modelSavePath, checkpoint=None):
    loss = 0
    model = model.to(device)
    epochStart = 0
    print(model)
    # load checkpoint
    if checkpoint is not None:
        epochStart = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    # train begin
    for epoch in range(epochStart, EPOCH):

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('epoch:{} , learining rate:{} , start'.format(epoch, lr))
        model.train()
        train_loss, train_acc = trainOneEpoch(
            model, trainloader, epoch, optimizer, device, None)
        # scheduler.step()
        # validate
        val_loss, val_acc = validate(model, testloader, device,
                                     state_dict=model.state_dict())
        print('----test epoch:{}, loss : {} , accuracy : {}, '.format(epoch, val_loss, val_acc))
        # save info
        tags = ["train_loss", "train_acc",
                "val_loss", "val_acc", "learning_rate"]
        tbWriter.add_scalar(tags[0], train_loss, epoch)
        tbWriter.add_scalar(tags[1], train_acc, epoch)
        tbWriter.add_scalar(tags[2], val_loss, epoch)
        tbWriter.add_scalar(tags[3], val_acc, epoch)
        tbWriter.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # save model
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch}
        if not os.path.exist(modelSavePath):
            os.mkdir(modelSavePath)
        torch.save(state, modelSavePath +
                   'checkpoint-epoch-{}.pth'.format(epoch))
    return model


if __name__ == '__main__':
    trainset = MineCars(path=DataSetRoot, labelpath='labels', imageSize=(1280, 720),
                        train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    testset = MineCars(path=DataSetRoot, labelpath='labels', imageSize=(1280, 720),
                       train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE * BATCH_SIZE, shuffle=True, num_workers=8)

    model = dense
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.3, last_epoch=-1)
    lossFn = nn.CrossEntropyLoss()

    # checkpoint = torch.load(model_save_root+'Vit/checkpoint-epoch-49.pth',map_location=torch.device(device))
    checkpoint = None
    modelSavePath = ModelSaveRoot
    trainInfoPath = TrainInfoRoot
    train(trainloader, testloader, model, optimizer, scheduler,
          device, modelSavePath, checkpoint)
    # correctRate = validate(model, testloader, device,
    #                        state_dict=checkpoint['net'])
    # print(correctRate)
