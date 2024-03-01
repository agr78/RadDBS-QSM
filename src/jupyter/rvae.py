__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import numpy as np
import datetime

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    # x = x.view(x.size(0), 1, 28, 28)
    return x

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class VAE(nn.Module):
    def __init__(self,X_train=None):
        super(VAE, self).__init__()
        self.X_train = X_train
        self.in_size = 2925
        self.yl = 1
        self.fc1 = nn.Linear(self.in_size*self.yl, self.in_size*self.yl//2)
        self.fc21 = nn.Linear(self.in_size*self.yl//2, self.in_size*self.yl//4)
        self.fc22 = nn.Linear(self.in_size*self.yl//2, self.in_size*self.yl//4)
        self.fc3 = nn.Linear(self.in_size*self.yl//4, self.in_size*self.yl//4)
        self.fc4 = nn.Linear(self.in_size*self.yl//4, self.in_size*self.yl//2)
        self.fc5 = nn.Linear(self.in_size*self.yl//2, self.in_size*self.yl)
        self.r = nn.Linear(self.in_size*self.yl,1)
        self.rl = nn.Linear(self.in_size*self.yl//4,1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        # z = σε + μ 
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        #return torch.sigmoid(self.fc5(h4))
        return self.fc5(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        yi = self.predict(x)
        yh = self.predict(self.decode(z))
        yhl = self.latent_predict(z)
        return self.decode(z), mu, logvar, yi, yh, yhl
    
    def predict(self,x):
        return self.sigmoid(abs(self.r(x)))
    
    def latent_predict(self,z):
        return self.sigmoid(abs(self.rl(z)))

model = VAE()

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: approximated image output
    x: original image input
    mu: latent mean
    logvar: latent log variance
    """
    reconstruction_function = nn.MSELoss(reduction='sum')
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

def train_model(X_train,y_train,model,X_test,case_idx):
    torch.autograd.set_detect_anomaly(True)
    num_epochs = 50
    model.cuda()
    mse = nn.MSELoss()#reduction='sum')
    Xpt = torch.zeros_like(torch.Tensor(X_test))
    lvst = torch.zeros((X_test.shape[0],X_test.shape[1]//2))
    Xp = torch.zeros_like(torch.Tensor(X_train))
    lvs = torch.zeros((X_train.shape[0],X_train.shape[1]//2))
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_train.cuda()
    y_train.cuda()
    ys = torch.zeros_like(y_train)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=10)
    best_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        # Assumes batch size of 1
        for case in np.arange(X_train.shape[0]):
            img = X_train[case,:]
            img = Variable(img).cuda()                
            optimizer.zero_grad()
            recon_batch, mu, logvar, yi, yh, yhl = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)/1000+mse(yh[0],y_train[case].cuda())+mse(yhl[0],y_train[case].cuda())+mse(yi[0],y_train[case].cuda())
            loss.backward()
            train_loss += loss
            optimizer.step()
            # lr_scheduler.step()

            # if epoch % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch,
            #         batch_idx,
            #         X_train.shape[0], 100. * batch_idx / X_train.shape[0],
            #         loss))
        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #     epoch, train_loss/X_train.shape[0]))
        # if epoch % 10 == 0:
        #     save = to_img(recon_batch.cpu().data)
        #     save_image(save, './vae_img/image_{}.png'.format(epoch))

    
    torch.save(model.state_dict(), './vae_img/net_'+str(case_idx)+'_'+str(datetime.datetime.now())+'.pth')
    model.eval()
    for case in np.arange(X_train.shape[0]):
        Xh = model(torch.Tensor(X_train[case,:]).cuda())
        Xp[case,:] = Xh[0]
        lvs[case,:] = torch.hstack((Xh[1],Xh[2]))
        ys[case] = Xh[3]
    Xht = model(torch.Tensor(X_test).cuda())
    Xpt = Xht[0]
    lvst = torch.hstack((Xht[1],Xht[2]))
    yi = Xht[3]
    yh = Xht[4]
    yht = Xht[5]
    return Xp, lvs, ys, Xpt, lvst, yi, yh, yht