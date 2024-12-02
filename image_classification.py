# This program is to finish the image task in homework in requirements of 
# scientific machine learning course
# written by fuyi li on Friday 2024 11 29

# import matplotlib.pylot as plt
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
import time 

class image_classification_net(nn.Module): # conherint class nn.Module and create new class
    def __init__(self, num_classes):
        super(image_classification_net, self).__init__()
        # convolution layer1, step length = 1, kernel size = 3, in/out channel = 1/32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # concolution layer construction
        
        # convolution layer2, step length = 2, kernel size = 3, in/out channel = 32/64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64*7*7, 10)
        self.fc2 = nn.Linear(10, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # view x
        x = x.view(-1, 64*7*7)

        # full connection layer
        x = F.relu(self.fc1(x))
        y_pred = self.fc2(x)
        
        return y_pred


# download dataset of minist and load in to buffer
start_time = time.time()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081, ))
])
dataset = datasets.MNIST('./minist_dataset', train=True, download=False, transform=transform)
total_size=  len(dataset)
train_size = int(0.8*total_size)
test_size = total_size - train_size
train_imageset, test_imageset = random_split(dataset, [train_size, test_size])
ex_image, ex_label = train_imageset[0]
print(ex_image.shape)

# create neural network 
net = image_classification_net(num_classes=10)
lossfunc = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# transform dataset into iterable data type
batch_size = 64
shuffle = True
train_loader = DataLoader(train_imageset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_imageset, batch_size=batch_size, shuffle=False)

# set epoches and start training
num_epoches = 10
for epoch in range(num_epoches):
    net.train() # set mode to train mode
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = lossfunc(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # loop test
    net.eval() # set mode to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        accuracy = 100*correct/total
        print(f'epoch [{epoch+1}/{num_epoches}], accuracy: {accuracy:.2f}%')

# direct show predicted result and true label
num_samples = 10
random_indices = random.sample(range(len(dataset)), num_samples)
samples = [dataset[i] for i in random_indices]

predictions = []
for image, label in samples:
    with torch.no_grad():
        output = net(image.unsqueeze(0)) 
        _, predicted = torch.max(output, 1)
        predictions.append(predicted.item())

mean = 0.1307
std = 0.3081
images = [(image * std + mean).squeeze().numpy() for image, _ in samples]
labels = [label for _, label in samples]

# plot handwritting digital number
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i in range(num_samples):
    ax = axes[i]
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"True: {labels[i]}, Pred: {predictions[i]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('./output/neural_network/image_classification_visualization.png', dpi=300)

end_time = time.time()
print('All figure classfication training has been done!')
print(f'Time cost is: {(end_time - start_time):.4f} s')
