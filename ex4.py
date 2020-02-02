import os
import os.path
import torchvision
from torchvision import transforms

import soundfile as sf
import librosa
import numpy as np
import torch
import torch.utils.data as data
from torch import nn, optim
import torch.utils.data
from gcommand_loader import GCommandLoader

dev = torch.device("cuda" if torch.cuda else "cpu")

PATH_TRAIN = "./data/train"
PATH_VALID = "./data/valid"
PATH_TEST = "./data/test"
FILE_TEST_Y = "test_y"
FIRST_NORM = 0.1307
SECOND_NORM = 0.3081

num_epochs = 7
val_eta = 0.001
size_batch = 100
amount_tag = 30

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((FIRST_NORM,), (SECOND_NORM,))])


class ConvolutionModel(nn.Module):
    def __init__(self):
        super(ConvolutionModel, self).__init__()
        self.firstLay = self.layer_builder(1, 16)
        self.secondLay = self.layer_builder(16, 64)
        self.thirdLay = self.layer_builder(64, 15)
        self.drop_out = nn.Dropout()
        self.lt1 = nn.Linear(3600, 150)
        self.lt2 = nn.Linear(150, 50)
        self.lt3 = nn.Linear(50, 30)

    def layer_builder(self, num1, num2):
        return nn.Sequential(
            nn.Conv2d(num1, num2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        fwd = self.firstLay(x)
        fwd = self.secondLay(fwd)
        fwd = self.thirdLay(fwd)
        fwd = fwd.reshape(fwd.size(0), -1)
        fwd = self.drop_out(fwd)
        fwd = self.lt1(fwd)
        fwd = self.lt2(fwd)
        fwd = self.lt3(fwd)
        return fwd


def func_train(m, loader_train, func_loss, opti):
    arr_loss = []
    arr_avg = []
    train_length = len(loader_train)
    for e in range(num_epochs):
        for i, (exa, tag) in enumerate(loader_train):
            exa, tag = exa.to(dev), tag.to(dev)
            arr_y_hat = m(exa)
            val_loss = func_loss(arr_y_hat, tag)
            arr_loss.append(val_loss.item())
            opti.zero_grad()
            val_loss.backward()
            opti.step()
            predictions = tag.size(0)
            _, predict = torch.max(arr_y_hat.data, 1)
            corr_pre = (predict == tag).sum().item()
            arr_avg.append(corr_pre / predictions)


def func_validation(m, loader_valid):
    m.eval()
    with torch.no_grad():
        predicts = 0
        predicts_arr = 0
        for exa, tag in loader_valid:
            exa, tag = exa.to(dev), tag.to(dev)
            arr_y_hat = m(exa)
            _, predict = torch.max(arr_y_hat.data, 1)
            predicts_arr += tag.size(0)
            predicts += (predict == tag).sum().item()


def func_test(m, loader_test, set_test):
    m.eval()
    arr_predicts = []
    with torch.no_grad():
        for tag, _ in loader_test:
            tag = tag.to(dev)
            _ = _.to(dev)
            arr_y_hat = m(tag)
            _, predict = torch.max(arr_y_hat.data, 1)
            arr_predicts.extend(predict)
        with open(FILE_TEST_Y, "w") as file:
            for s, pre in zip(set_test.spects, arr_predicts):
                file.write("{}, {}".format(os.path.basename(s[0]), str(pre.item()) + '\n'))


def func_load_data(add, shuffle):
    load = GCommandLoader(add)
    return torch.utils.data.DataLoader(load, batch_size=100, shuffle=shuffle,
                                       num_workers=20, pin_memory=True, sampler=None)


if __name__ == '__main__':
    loader_train = func_load_data(PATH_TRAIN, True)
    loader_valid = func_load_data(PATH_VALID, True)
    loader_test = func_load_data(PATH_TEST, False)
    m = ConvolutionModel().to(dev)
    func_loss = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(m.parameters(), lr=val_eta)
    func_train(m, loader_train, func_loss, opti)
    func_validation(m, loader_valid)
    func_test(m, loader_test, GCommandLoader(PATH_TEST))
