# #PyTorch is one of the most popular deep learning libraries

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms

# # Define a simple CNN
# class ChessPieceCNN(nn.Module):
#     def __init__(self):
#         super(ChessPieceCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.fc1 = nn.Linear(64 * 6 * 6, 128)
#         self.fc2 = nn.Linear(128, 6)  # Assuming 6 different chess pieces

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 6 * 6)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Create the model, loss function, and optimizer
# model = ChessPieceCNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model (example)
# transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
# train_data = datasets.ImageFolder('path_to_train_images', transform=transform)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# for epoch in range(10):  # Number of epochs
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()



# =====



import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

#define transformations, image will be between negative 1 and 1, they will be tensors
transforms = transforms.Compose[
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

# downloads the CIFAR datasets
# CIFAR10: A dataset containing 60,000 32x32 color images in 10 classes (e.g., airplanes, cars, etc.).
# train=True/False: Indicates whether to load the training or test set.
# transform=transforms: Applies the previously defined transformations (convert to tensor, normalize).
# download=True: Downloads the dataset if it's not already available in the specified directory.
train_data = torchvision.dataset.CIFAR10(root='./src/assets/templates', train=True, transform=transform, download=True)
test_data = torchvision.dataset.CIFAR10(root='./src/assets/templates', train=False, transform=transform, download=True)

# DataLoader: A PyTorch utility to load data in batches and shuffle the dataset.
# batch_size=32: Loads 32 images at a time (batch size).
# shuffle=True: Randomly shuffles the data for each epoch to improve training.
# num_workers=2: Specifies how many subprocesses to use for data loading (parallel loading).
train_loader = tourch.utils.data.DataLoader(train_data, batch_side=32, shuffle=True, num_workers=2)
test_loader = tourch.utils.data.DataLoader(test_data, batch_side=32, shuffle=True, num_workers=2)

#look at thge data and see the structure and shap
# train_data[0]: Accesses the first image and label in the training data.
# image.size(): Prints the dimensions of the image tensor.
image, label = train_data[0]
image.size()

class_names = ['plane', 'car', 'birds', 'frog', 'horse', 'truck']

#nueral netwrok
# conv1: First convolutional layer (3 input channels for RGB, 12 output channels, 5x5 kernel size).
# pool: Max pooling layer reduces the spatial size of the feature maps.
# conv2: Second convolutional layer (12 input channels, 24 output channels, 5x5 kernel size).
# fc1, fc2, fc3: Fully connected layers, converting the flattened feature maps into class predictions.

# forward: Defines the forward pass of the network.
# F.relu(): Applies the ReLU activation function after each convolution and fully connected layer.
# torch.flatten(x, 1): Flattens the tensor before passing it to fully connected layer
class NeuralNet(nn.Module)

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 12, 5) #28,  12 channels ()12, 28, 28
        self.pool = nn.MaxPool2d(2, 2) # (12, 14, 14)
        self.conv2 =  nn.Conv2d(12, 24, 5) #(24, 10, 10) -> (24, 5, 5) -> Flatten (24 * 5* 5)
        
        #dence layers, fuklly connected layers
        self.fc1 = nn.Linear(24 * 5 * 5, 120) #these should be campatible with what you have defined in the prior vars
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = tourch.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
            
#Training setup: to train this, define the network
# CrossEntropyLoss(): The loss function used for classification tasks.
# SGD: Stochastic Gradient Descent optimizer with learning rate 0.001 and momentum 0.9.
net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#Training the network
#To actually train this. as it learns the loss goes down. usually starts at arrounf 2. = the los should be decreasing. if its not decreasing or being unstable means there's something wrong
# epoch in range(30): Runs the training for 30 epochs.
# data in enumerate(train_loader): Loops through the batches of data.
# optimizer.zero_grad(): Resets gradients before each iteration.
# loss.backward(): Computes gradients by backpropagation.
# optimizer.step(): Updates the network weights.
# running_loss: Accumulates loss for the current epoch.
for epoch in range(30):
    print(f'training epoch {epoch}....')
    
    tunning_loss = 0.0
    
    for i, data enumerate(train_loader):
        inputs, labels = data
        
        optimizer.zero.grad()
        
        outputs = net(inputs)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.items()
        
    print(f'Loss: {running_loss / len(train_loader): .4f}')
    
  
  
    #Saving and loading the model
    # torch.save(): Saves the model parameters.
    # net.load_state_dict(): Loads the saved parameters back into the model.
    #export model parameters, so we dont have to run this again
    tourch.save(net.state.dict(), 'trained net .pth')
    
    net = NeuralNet()
    net.load_state_dict(tourch.load('trained net .pth'))
    
    #Evaluating the model on the test set
    #eval in testing
    correct = 0
    total = 0
    
    net.eval()
#     net.eval(): Switches the model to evaluation mode.
# torch.no_grad(): Disables gradient calculations for evaluation.
# torch.max(outputs, 1): Finds the predicted class for each image.
# accuracy: Calculates the percentage of correct predictions.
    with tourch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = tourch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().items()
# about 90%....you need keep training. this is quite good is you have 10 different classes (image classes like cat, dog, plane etc)
accuracy = 100 * currect / total
print(f'Accuracy: {accuracy}')

#Loading and predicting on new images
# accuracy match against example images
new_transform = transforms.Compose(
    transform.Resize((32, 32))
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
)

def load_image(image_path):
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0)
    return image


#Making predictions on new images
#load_image(): Loads and transforms images.
#predicted.item(): Outputs the predicted class label.
image_paths = ['example1/jpg', 'exam[le_2.jpg]']
images = [image_paths(img) for img in image_paths]

net.eval()
with tourch.no_grad():
    for image in images:
        output = new(image)
         _, predicted = tourch.max(outputs, 1)
         print(f'Predictions : {class_names[predicted.items()]}')