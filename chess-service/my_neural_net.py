import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
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
    

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)  # Adjust output to match number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    print("Main block is running")
    
    # Early stopping parameters
    early_stopping_patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Define transformations with various augmentations
    transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.RandomResizedCrop((110, 110), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        RandomCutout(mask_size=10),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download CIFAR-10 dataset
    train_data = torchvision.datasets.CIFAR10(root='./src/assets/templates', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR10(root='./src/assets/templates', train=False, transform=transform, download=True)

    # DataLoader for GPU
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Move the model to the selected device
    net = NeuralNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
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

    if evaluate_model:
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

print(f"The value of __name__ is: {__name__}")
# on certain operating systems (especially Windows and macOS), the multiprocessing library requires special handling when starting new processes (e.g., when using num_workers > 0 in a DataLoader). Specifically, if code is run in a way that spawns multiple processes (such as a DataLoader with num_workers), Python needs to know which part of the code to run only in the main process. This is done using the if __name__ == '__main__' block.
# IMPORTANT!!! RUN FILE DIRECTLY ON DEVELOPMENT MODE - "python my_neural_net.py"
if __name__ == '__main__':
    main()
