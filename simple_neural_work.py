# this program is written to solve non-linear regression using nerual network 
# written by fuyi li date: 2024 11 28

import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from sklearn.model_selection import train_test_split
import time 
# import torchvision.transforms as transform

np.random.seed(1) # set global random seed

class Net(nn.Module): # define nerual network containing two networks
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #input to hidden
        self.relu = nn.ReLU() #nonlinear activate
        self.fc2 = nn.Linear(hidden_size, output_size) #hidden to output
    
    def forward(self, x):
        x = self.fc1(x) # doing forward calculate
        x = self.relu(x)
        y = self.fc2(x)
        return y

def real_func(x):
    # return x
    return x**2
    # return np.sin(10*math.pi*x)

# generate data for training
start_time = time.time()
x_left_limit = -1
x_right_limit = 1
generated_dots_num = 1000
x = np.random.uniform(x_left_limit, x_right_limit, (generated_dots_num,1)) 
y = real_func(x)
x = torch.from_numpy(x).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# construct nerual network for training
input_size = 1
output_size = 1
hidden_size = 40 # this refers to D in this homework
net = Net(input_size, hidden_size, output_size)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # define optimizer method
lossfunc = nn.MSELoss()
print(net)

# setting num_epoches
num_epoches = 1000
for epoch in range(num_epoches):
    # forward spread
    y_pred = net(x_train)
    
    # loss calculate
    loss = lossfunc(y_pred, y_train)

    # grad calculate and backward spread
    optimizer.zero_grad()
    loss.backward()

    # update parameters
    optimizer.step()

    if((epoch+1)%100==0):
        print(f'epoch [{epoch+1}/{num_epoches}], Loss: {loss.item():.8f}')

with torch.no_grad():
    y_pred = net(x_test)
    pred_loss = lossfunc(y_pred, y_test)
    print(f'test loss: {pred_loss.item():.8f}')

# plot figure to direct show result
x_test = x_test.numpy()[::5]
y_test = y_test.numpy()[::5]
y_pred = y_pred.numpy()[::5]
plt.plot(x_test, y_test, 'x', color='blue', alpha=0.8, label='real value')
plt.plot(x_test, y_pred, '.', color='red', alpha=0.3, label='pred value')
plt.legend(loc='best')
plt.xlabel('x axis')
plt.ylabel('y axis')
# plt.title(r'$y=\sin(10 \pi x)$')
# plt.title(r'$y=x$')
plt.title(r'$y=x^2$')
plt.savefig('./output/neural_network/y_x2_'+'D_'+str(hidden_size)+'.png', dpi=300)
# plt.savefig('./output/neural_network/y_sin(10pi_x)_'+'D_'+str(hidden_size)+'.png', dpi=300)
end_time = time.time()
print('All the training process has been done!')
print(f'Time cost is: : {(end_time - start_time):.4f} s')