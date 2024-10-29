import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

CATEGORY_CLASSES = 7
SQUARE_WIDTH = 135 # 224
SQUARE_HEIGHT = 135 # 224
# Define your class names
CLASS_NAMES = ['bishop', 'empty', 'king', 'knight', 'pawn', 'queen', 'rook']



# The RandomCutout transform is a form of data augmentation where a small, randomly located rectangular section of an image is masked (set to zero). This technique can help your neural network become more robust by forcing it to rely on different parts of the image to make predictions rather than focusing on a particular area.
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
        
        # Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),  # Convolution
            nn.ReLU(),
            nn.MaxPool2d(3, 2)    # Max pooling
        )
        
        # Layer 2
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3))
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(3, 2)
        
        # Calculate flattened size automatically
        # Run a dummy input through conv layers
        dummy_input = torch.zeros(1, 3, 135, 135)
        dummy_output = self._forward_conv_layers(dummy_input)
        flattened_size = dummy_output.view(-1).size(0)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, CATEGORY_CLASSES)
    
    def _forward_conv_layers(self, x):
        x = self.conv1(x)
        x = self.relu_2(self.conv2d_2(x))
        x = self.maxpool_2(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


class NeuralNet_old(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Layer 1
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32,3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3, 2))
        # Layer 2
        self.conv2d_2 = nn.Conv2d(32, 64, (3,3))
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(3, 2)
        # Layer 3
        self.fc1 = nn.Linear(179776, 32)
        self.fc2 = nn.Linear(32, CATEGORY_CLASSES)
        
        # self.conv1 = nn.Conv2d(3, 12, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(12, 24, 5)
        # self.fc1 = nn.Linear(24 * 24 * 24, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 7)  # Adjust output to match number of classes
        
        # # Convolutional and Pooling Layers
        # self.conv1 = nn.Conv2d(3, 12, kernel_size=5)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(12, 24, kernel_size=5)
        # # Calculate the flattened dimension after conv2 and pool
        # # Input: 224x224 -> Conv1: 220x220 -> Pool: 110x110
        # # Conv2: 106x106 -> Pool: 53x53
        # flattened_dim = 24 * 53 * 53
        # # Fully Connected Layers
        # self.fc1 = nn.Linear(flattened_dim, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 7)  # Output layer for 7 classes
        
        # # Layer 1
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(3, 2)
        # )
        # # Layer 2
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(3, 2)
        # )
        # # Additional Layer 3
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(3, 2)
        # )
        # # Fully Connected Layers
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(128 * 25 * 25, 64)  # Adjusted for calculated dimension
        # self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        # Layer 1
        y = self.conv1(x)
        # Layer 2
        y = self.conv2d_2(y)
        y = self.relu_2(y)
        y = self.maxpool_2(y)
        # Layer 3
        y = y.view(y.size(0), -1) 
        y = self.fc1(y)
        y = self.fc2(y)
        return y
    
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        
        # x = self.pool(nn.ReLU()(self.conv1(x)))
        # x = self.pool(nn.ReLU()(self.conv2(x)))
        # x = x.view(-1, 24 * 53 * 53)  # Flatten
        # x = nn.ReLU()(self.fc1(x))
        # x = nn.ReLU()(self.fc2(x))
        # x = self.fc3(x)
        # return x
        
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = nn.ReLU()(x)
        # x = self.fc2(x)
        # return x
        
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
# Initialize a global variable to store classes
_global_classes = None

# Setter function to set classesdef set_classes(train_data):
def set_classes(train_data):
    global _global_classes
    if hasattr(train_data, 'classes'):
        _global_classes = train_data.classes
        print("Classes set successfully:", _global_classes)
    else:
        raise AttributeError("train_data does not have a 'classes' attribute")


# Getter function to get classes
def get_classes():
    if _global_classes is not None:
        return _global_classes
    else:
        return CLASS_NAMES
 
def main():
    print("Main block is running")
    
    # Early stopping parameters
    early_stopping_patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    batch_size = 8 #64
    step_size = 7 #10
    
    epoch_max = 40

    # Define transformations with various augmentations
    transform = transforms.Compose([
        transforms.Resize([SQUARE_WIDTH ,SQUARE_HEIGHT]),
        # transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=15),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        # RandomCutout(mask_size=10),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        # consider more extra augmentation
    #     transforms.Resize([224, 224]),
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    from_google_drive = False
    #google drive is accesible only on dev mode, not locally
    if from_google_drive:
        from google.colab import drive
        drive.mount('/content/drive')
        # Set the path to your dataset in Google Drive
        train_data_path = '/content/drive/My Drive/dataset/train'
        test_data_path = '/content/drive/My Drive/dataset/test'
    # for local scan
    else:
        train_data_path = '/Users/sabina.livny/Pictures/dataset'
        test_data_path = '/Users/sabina.livny/Pictures/dataset'
    

    # ImageFolder and CIFAR10 in torchvision.datasets are both dataset classes.
    #ImageFolder:
    # Used to load images from a directory structure on your local system.
    # It expects a specific directory format, where each class has its own subfolder. The name of each subfolder is treated as the label for that class.
    # Ideal for custom datasets that are organized manually in folders.
    # It doesn’t download anything and assumes the images are already on your system.
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)

    # Loading Class Names Dynamically (from a folder structure): If you're using torchvision.datasets.ImageFolder to load images, it will automatically assign train_data.classes based on the folder names. This works if your data folder structure is
    set_classes(train_data)
    print("Class names:", get_classes())  # Check that classes are defined
    
    # Download CIFAR-10 dataset
    #     CIFAR10:
    # This is a pre-defined dataset class specifically for the CIFAR-10 dataset, which is a popular image dataset for machine learning benchmarks.
    # CIFAR10 automatically downloads the CIFAR-10 dataset if it’s not already available in the specified directory (through the download=True argument).
    # The dataset has 10 classes, and the labels are automatically included (you don’t need to specify subfolders for each class).
    # train_data = torchvision.datasets.CIFAR10(root=train_data_path, train=True, transform=transform, download=True)
    # test_data = torchvision.datasets.CIFAR10(root=test_data_path, train=False, transform=transform, download=True)

    # DataLoader for GPU
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Device setup
    if torch.backends.mps.is_available():  # For Apple Silicon GPUs (macOS)
        device = torch.device("mps")
    elif torch.cuda.is_available():  # For NVIDIA GPUs on Linux/Windows
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")  # Default to CPU

    # Move the model to the selected device
    net = NeuralNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)  # Adjusted step_size

    manual_train = True  # Set this to True to trigger training

    if manual_train:
        for epoch in range(epoch_max):
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
    compute_confusion_matrix = False
    
    # List of class names
    class_names = train_data.classes 
    
    if evaluate_model:
        net.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if compute_confusion_matrix:
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    
        
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

        if compute_confusion_matrix:
            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_predictions)
            plot_confusion_matrix(cm, class_names)

# print(f"The value of __name__ is: {__name__}")
# on certain operating systems (especially Windows and macOS), the multiprocessing library requires special handling when starting new processes (e.g., when using num_workers > 0 in a DataLoader). Specifically, if code is run in a way that spawns multiple processes (such as a DataLoader with num_workers), Python needs to know which part of the code to run only in the main process. This is done using the if __name__ == '__main__' block.
# IMPORTANT!!! RUN FILE DIRECTLY ON DEVELOPMENT MODE - "python my_neural_net.py" - you can also try keeping this code under main on Linux
if __name__ == '__main__':
    main()
