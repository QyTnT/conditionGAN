import numpy as np
import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, in_size, out_size, norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = nn.Linear(in_size,out_size)
        self.bn = nn.BatchNorm1d(out_size,0.8)
        self.act = nn.LeakyReLU(0.2,inplace=False)
        self.norm = norm
    def forward(self,x):
        x = self.layer(x)
        if self.norm:
            x = self.bn(x)
        x = self.act(x)
        return self.bn(x)

#生成器
class Generator(nn.Module):
    def __init__(self,numclasses, latent_dim, img_shape):
        super(Generator,self).__init__()

        self.label_emb = nn.Embedding(numclasses,numclasses)
        self.model = nn.Sequential(
            Block(numclasses+latent_dim,128,norm=False),
            Block(128,256),
            Block(256,512),
            Block(512,1024),
            nn.Linear(1024,int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self,noise,labels):
        gen_img = torch.cat((self.label_emb(labels),noise),-1)
        img = self.model(gen_img)
        img = img.view(img.size(0),*self.img_shape)
        return img


#判别器
class Discriminator(nn.Module):
    def __init__(self, numclasses, img_shape):
        super(Discriminator,self).__init__()

        self.label_emb = nn.Embedding(numclasses, numclasses)
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(numclasses + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        validity = self.model(d_in)
        return validity