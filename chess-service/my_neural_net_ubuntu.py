import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class RandomCutout:
    """Custom transform for randomly masking out sections of an image"""
    def __init__(self, mask_size):
        self.mask_size = mask_size

    def __call__(self, img):
        # Choose random mask location
        mask_x = random.randint(0, img.size(1) - self.mask_size)
        mask_y = random.randint(0, img.size(2) - self.mask_size)
        
        # Apply mask
        img[:, mask_x:mask_x + self.mask_size, mask_y:mask_y + self.mask_size] = 0
        return img
    

    
# Early stopping parameters
early_stopping_patience = 3  # Stop after 3 epochs with no improvement
best_val_loss = float('inf')  # Best validation loss so far
epochs_no_improve = 0  # Count how many epochs with no improvement

# Define transformations for grayscale images (110x110, between -1 and 1)
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
#     transforms.Resize((110, 110)),  # Resize to 110x110
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for grayscale
# ])
# Define transformations for RGB images (110x110, between -1 and 1)
# transform = transforms.Compose([
#     transforms.Resize((110, 110)),  # Resize to 110x110
#         transforms.Grayscale(num_output_channels=3),  # Ensure grayscale images are replicated across RGB channels
#             # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Augment color images
#     transforms.ToTensor(),# Convert to tensor
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] for RGB
# ])

# Define transformations with various augmentations
transform = transforms.Compose([
    transforms.Resize((120, 120)),  # Initially resize slightly larger for cropping
    transforms.RandomResizedCrop((110, 110), scale=(0.8, 1.0)),  # Random crop and scale
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=15),  # Small random rotation
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Color jitter
    transforms.ToTensor(),  # Convert to tensor
    RandomCutout(mask_size=10),  # Apply cutout with a mask size of 10x10
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Download CIFAR-10 dataset (for example purposes)
train_data = torchvision.datasets.CIFAR10(root='./src/assets/templates', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./src/assets/templates', train=False, transform=transform, download=True)

# for CPU
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

# for GPU
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # 1: This is the number of input channels. Since you're using grayscale chess images, you set this to 1. If your images were RGB (color), you'd use 3 (for 3 channels: Red, Green, Blue).
        # 12: This is the number of output channels (or feature maps). This is arbitrary, but a typical choice for the first convolutional layer is between 8 and 32 channels. More channels allow the network to learn more complex features.
        # 5: This is the size of the convolution filter (kernel). A 5x5 filter is a common size used to detect spatial features in images. Other common choices are 3x3 or 7x7. Smaller filters capture fine details, while larger ones capture broader patterns.
        self.conv1 = nn.Conv2d(3, 12, 5)
        #2, 2: This is the size of the pooling window (2x2) and the stride (how much the window shifts between applications). Max pooling reduces the spatial dimensions (height and width) of the input by taking the maximum value in each 2x2 region. Pooling reduces computation and helps prevent overfitting by discarding less important features.
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half
        # 12: This is the number of input channels for the second convolution. It must match the output channels of the first layer.
        # 24: This is the number of output channels (again, it's arbitrary but commonly increased compared to the first layer).
        # 5: The size of the convolution filter is again 5x5.
        self.conv2 = nn.Conv2d(12, 24, 5)
        # 4. Fully Connected Layers (Linear)
        #  start with the size of the original image (let's assume it's 32x32).
        # After the first convolution (5x5 kernel), the output feature map size will reduce to 28x28 (because with no padding, 32 - 5 + 1 = 28).
        # Max pooling reduces this by half to 14x14.
        # After the second convolution (another 5x5 kernel), the feature map size will reduce to 10x10 (because 14 - 5 + 1 = 10).
        # Max pooling reduces this by half to 5x5.
        # So after two convolutions and two pooling layers, your feature map is 24 channels of size 5x5. Therefore, the total number of features is 24 * 5 * 5 = 600
        self.fc1 = nn.Linear(24 * 24 * 24, 120)
        #Second Fully Connected Layer
        # 120: This is the output of the previous layer.
        # 84: You can adjust the size here. It's often a smaller number than the first fully connected layer, used to further reduce dimensionality.
        self.fc2 = nn.Linear(120, 84)
        # Output Layer
        # 84: This is the output from the previous layer.
        # 12: This should match the number of classes you're classifying. In your code, it is set to 10, which matches CIFAR-10 dataset (10 classes). For a chess game, you might want to change this to match the number of classes (e.g., 12 classes if you include 6 white and 6 black pieces).
        self.fc3 = nn.Linear(84, 12)  # 10 output classes (for CIFAR10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training setup
# Check if CUDA (GPU) is available. GPU is faster than CPU training - for Ubuntu Linux
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the selected device (CPU or GPU)
net = NeuralNet().to(device)
#net = NeuralNet()
loss_function = nn.CrossEntropyLoss()

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Stochastic Gradient Descent (SGD) with a static learning rate may converge slowly. You can switch to an optimizer like Adam or RMSProp, which adapt the learning rate during training and often result in faster convergence and better performance.
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
#The learning rate may be too high or too low for different stages of training. You can introduce a learning rate scheduler to reduce the learning rate as training progresses, which helps with fine-tuning later epochs and avoiding getting stuck in local minima.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)




manual_train = True  # Set this to True to trigger training

if manual_train:
    for epoch in range(30):
        print(f'Training epoch {epoch}....')
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Training Loss: {avg_train_loss:.4f}')

        # Validation phase
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(net.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            break

        # Save model every epoch (optional)
        torch.save(net.state_dict(), 'trained_net.pth')

        # Learning rate scheduler step
        scheduler.step()
else:
    print("Training skipped. Set manual_train to True to start training.")
    
    
# Manual flag for evaluating the model
evaluate_model = True  # Set to False to skip evaluation

# Evaluating the model (only runs if evaluate_model is True)
if evaluate_model:
    # Evaluating the model
    net.eval()
    correct = 0
    total = 0
    #with is used for context management. It is a syntax that allows you to wrap the execution of a block of code within methods defined by a context manager, ensuring that certain actions are taken before and after the block of code is executed. The most common use of with is for managing resources like files, network connections, or locking mechanisms in concurrent programming.
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}')
