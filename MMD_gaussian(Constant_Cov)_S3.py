#### torch.manual_seed(0)
import pandas as pd
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torchvision.datasets
# import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.linear_model import LassoCV
import time
from layers import SinkhornDistance
from scipy.stats import wasserstein_distance
import pickle
from scipy import stats
import matplotlib as mpl
device = torch.device("cpu")


def MMD(x, y, brange):

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    bandwidth_range2 = brange
    for a in bandwidth_range2:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


class Phi(torch.nn.Module):  # Define Transformation Phi
    def __init__(self, n, d, hidden_layer_size):
        super(Phi, self).__init__()
        self.w = nn.Parameter(torch.randn(hidden_layer_size, 1).float())

        self.b = nn.Parameter(torch.randn(hidden_layer_size, 1).float())

        self.w2 = nn.Parameter(torch.randn(hidden_layer_size, hidden_layer_size).float())
        self.b2 = nn.Parameter(torch.randn(hidden_layer_size, 1).float())

        self.w3 = nn.Parameter(torch.randn(hidden_layer_size, hidden_layer_size).float())
        self.b3 = nn.Parameter(torch.randn(hidden_layer_size, 1).float())

        self.u = nn.Parameter(torch.randn(hidden_layer_size, 1).float())
        self.c = nn.Parameter(torch.tensor(1).float())

        self.size = n * d

        self.n = n
        self.d = d
        self.ones = torch.ones(1, self.size)
        self.m = nn.LeakyReLU(0.2)
        self.m2 = nn.LeakyReLU(0.2)
        self.m3 = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = torch.mm(self.w, torch.transpose(x, 0, 1)) + torch.mm(self.b, self.ones)

        h = self.m(z)
        z2 = torch.mm(self.w2, h) + torch.mm(self.b2, self.ones)
        h2 = self.m2(z2)

        _phi = torch.mm(self.u.T, h2) + self.c * self.ones

        return _phi.view(self.n, self.d)

    def test_forward(self, x):
        n_test = x.size()[0]
        d_test = x.size()[1]

        _test_size = n_test * d_test

        x = x.view(-1, 1)
        new_ones = torch.ones(1, _test_size)
        z = torch.mm(self.w, torch.transpose(x, 0, 1)) + torch.mm(self.b, new_ones)
        h = self.m(z)
        z2 = torch.mm(self.w2, h) + torch.mm(self.b2, new_ones)
        h2 = self.m2(z2)
        _phi = torch.mm(self.u.T, h2) + self.c * new_ones
        return _phi.view(n_test, d_test)


ans = []

# for batch_num in range(1):
m = 20  # num of nodes per layer Phi
m_datasets = 5
n_sample = 100
d = 101
num_iteration = 50000
ratio = 1  # train:(train+test)
phi = []
Transformation = []
W_distance = []
Z_torch = []  # for training
Z_torch_test = []
PhiOptimizers = []

train_size = int(ratio * n_sample)
test_size = n_sample - train_size
sigma = [0.01, 0.1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]

A = np.array([])
for i in range(d - 1):
    A = np.append(A, np.random.normal(0, 1, size=n_sample))
A = A.reshape(d - 1, n_sample)
A = A.T
A = torch.from_numpy(A).float()
Corr = torch.normal(0, 1, size=(d - 1, d - 1), requires_grad=True)
AZ = torch.matmul(A, Corr)
AZ = (AZ - torch.mean(AZ, dim=0)) / torch.std(AZ, dim=0)

tradeoff = 1e3
lam = 1e1

for i in range(m_datasets):
    name = 'Scenario3/data_y_transformed' + str(i + 1) + '.csv'
    data_read = pd.read_csv(name, header=None)  # read the datasets iteratively

    Z_train = data_read[:train_size]
    Z_test = data_read[train_size:]

    Z_train = Z_train.to_numpy()
    Vectorized_Z = Z_train.reshape(-1, 1)
    Z = torch.from_numpy(Vectorized_Z).float()
    Z_torch.append(Z)

    Z_test = Z_test.to_numpy()
    # Vectorized_Ztest = Z_test.reshape(-1, 1)
    Z_test = torch.from_numpy(Z_test).float()
    Z_torch_test.append(Z_test)

theta = torch.normal(0, 1, (d - 1, 1), requires_grad=True)  # random
y_fake = torch.matmul(AZ, theta)
data_fake = torch.cat((AZ, y_fake), -1)

for i in range(m_datasets):
    # train_size = all_data[i].shape[0]
    phi.append(Phi(train_size, d, m))
    transformation = phi[i](Z_torch[i])
    Transformation.append(transformation)
    W_distance.append(MMD(Transformation[i], data_fake, sigma))  # Y is not transformed

for i in range(m_datasets):
    PhiOptimizers.append(torch.optim.Adam(phi[i].parameters(), lr=1e-2))
ThetaOptimizer = torch.optim.Adam([theta], lr=5e-2)

prediction_array = []
regularization_array = []
l1_array = []
total_array = []
min_mmd = float('inf')
for j in range(num_iteration):

    for opt in PhiOptimizers:
        opt.zero_grad()
    ThetaOptimizer.zero_grad()

    for i in range(len(Transformation)):
        Transformation[i] = phi[i](Z_torch[i])

    y_fake = torch.matmul(AZ, theta)
    data_fake = torch.cat((AZ, y_fake), -1)

    for i in range(m_datasets):
        W_distance[i] = MMD(Transformation[i], data_fake, sigma)

    regularization = 0

    for i in range(m_datasets):
        regularization += W_distance[i]

    prediction_loss = 0
    for i in range(m_datasets):
        prediction_loss += torch.norm(torch.mm(Transformation[i][:, :-1], theta) - Transformation[i][:, -1],
                                      2) ** 2 / (m_datasets*n_sample)

    l1_norm = torch.norm(theta, 1)
    loss = prediction_loss + tradeoff*regularization + lam*l1_norm
    # loss = regularization
    #     loss = lam*l1_norm

    loss.backward(retain_graph=True)

    regularization_array.append(float(regularization))
    total_array.append(float(loss))
    l1_array.append(float(l1_norm))
    prediction_array.append(float(prediction_loss))

    for opt in PhiOptimizers:
        opt.step()
    ThetaOptimizer.step()

    for i in range(m_datasets):
        sss = 0
        for p in phi[i].parameters():
            #                 print(p.grad)
            if sss % 2 == 0:
                p.data = torch.clamp(p.data, min=0)
            sss += 1

    if j % 100 == 0:
        #             print(Transformation[0])
        print(j)
        print(regularization)
        print(prediction_loss)
        print(l1_norm)
        print("----")
    if (j + 1) % 1000 == 0:
        final_data = torch.cat((Transformation[0], Transformation[1]), 0)
        for i in range(2, m_datasets):
            final_data = torch.cat((final_data, Transformation[i]), 0)
        final_data = final_data.detach().numpy()
        final_data = pd.DataFrame(final_data)
        final_data.to_csv('final_data_3_' + str((j + 1) // 1000) + '.csv')


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(prediction_array)
axs[0, 0].set_title('prediction')
axs[0, 1].plot(l1_array)
axs[0, 1].set_title('l1')
axs[1, 0].plot(regularization_array)
axs[1, 0].set_title('regularization')
axs[1, 1].plot(total_array)
axs[1, 1].set_title('Total_Loss')
fig.tight_layout()
plt.show()




