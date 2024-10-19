import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

# Define transformations for grayscale images (110x110, between -1 and 1)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((110, 110)),  # Resize to 110x110
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for grayscale
])

# Download CIFAR-10 dataset (for example purposes)
train_data = torchvision.datasets.CIFAR10(root='./src/assets/templates', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./src/assets/templates', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)  # Grayscale image with 1 input channel, 12 output channels, 5x5 filter size
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes (for CIFAR10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training setup
# Check if CUDA (GPU) is available. GPU is faster than CPU training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the selected device (CPU or GPU)
net = NeuralNet().to(device)
#net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

manual_train = True  # Set this to True to trigger training

if manual_train:
    # Training loop
    for epoch in range(30):
        print(f'Training epoch {epoch}....')
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Loss: {running_loss / len(train_loader):.4f}')

        # Save model every epoch (optional)
        torch.save(net.state_dict(), 'trained_net.pth')
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
