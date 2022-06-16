import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, nc * 28 * 28)

        self.nc = nc

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        x = x.reshape((-1, self.nc, 28, 28))
        return x


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, loss):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)

        self.fc4 = nn.Linear(self.fc3.out_features, 1)

        if loss == "kl":
            # for KL - discriminator is a classifier
            self.act = torch.sigmoid
        else:
            # for Wasserstein and hinge - discriminator is a critic
            self.act = lambda x: x

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = self.act(self.fc4(x))
        return x
