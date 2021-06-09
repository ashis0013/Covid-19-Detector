from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np


def main(req):
    return render(req,'index.html',{})

def result(req):
    class SinLU(nn.Module):
        def __init__(self):
            super(SinLU,self).__init__()
            self.a = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.ones(1))
        def forward(self,x):
            return torch.sigmoid(x)*(x+self.a*torch.sin(self.b*x))

    class Squeeze(nn.Module):
        def __init__(self,in_ch,reduced_factor):
            super(Squeeze,self).__init__()
            self.in_ch = in_ch
            self.reduced = int(in_ch/reduced_factor)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch,self.reduced,1,1,0),
                SinLU(),
                nn.Conv2d(self.reduced,in_ch,1,1,0),
                nn.Sigmoid()
            )

        def forward(self,x):
            x *= self.se(x)
            return x

    class CNNBlock(nn.Module):
        def __init__(self,in_ch,out_ch):
            super(CNNBlock,self).__init__()
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.net = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,1,0),
                nn.BatchNorm2d(out_ch),
                SinLU(),
            )
            self.se = Squeeze(out_ch,2)
            self.acti = SinLU()

        def forward(self,x):
            identity = x
            x = self.net(x)
            x = self.se(x)
            x += identity
            return self.acti(x)


    class Net(nn.Module):
        def __init__(self,trans):
            super(Net,self).__init__()
            self.conv1 = nn.Sequential(
                trans.conv1,
                trans.bn1,
                trans.relu,
                trans.maxpool,
                trans.layer1,
            )
            self.block1 = CNNBlock(256,256)
            self.conv2 = trans.layer2
            self.block2 = CNNBlock(512,512)
            self.conv3 = trans.layer3
            self.block3 = CNNBlock(1024,1024)
            self.conv4 = trans.layer4
            self.block4 = CNNBlock(2048,2048)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Sequential(
                nn.Linear(3840,3),
            )

        def forward(self,x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x4 = self.conv4(x3)
            x1 = self.pool(self.block1(x1))
            x2 = self.pool(self.block2(x2))
            x3 = self.pool(self.block3(x3))
            x4 = self.pool(self.block4(x4))
            x = torch.cat((x1,x2,x3,x4),1)
            x = x.view(-1,3840)
            x = self.fc(x)
            return x
    def predict(img):
        T = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = T(img).view(-1,3,224,224)
        net = torch.load('best.pth',map_location=torch.device('cpu'))
        x = net(img).detach().numpy()[0]
        arr = np.exp(x) / np.sum(np.exp(x), axis=0)
        return np.multiply(np.round(arr,4),100)
    
    if req.method == 'POST':
        img = Image.open(req.FILES['bruh']).convert('RGB')
        res = predict(img)
        msg = ''
        if np.argmax(res) == 0: msg = 'You probably have COVID :('
        elif np.argmax(res) == 1: msg = 'You may have bacterial or viral pneuomonia, but it is not COVID.'
        else: msg = 'Bruh go have sex'
        ctx = {
            'covid':res[0],
            'pneu':res[1],
            'norm':res[2],
            'pred': msg
        }
        return render(req,'result.html',ctx)
    return render(req,'index.html',{})