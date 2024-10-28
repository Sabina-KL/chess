from piece_factory import PieceFactory
import numpy as np
from PIL import Image
from itertools import product
import os  # Import the os module
import torch
import torchvision.transforms as transforms
from my_neural_net import NeuralNet  # Import the pre-trained network

# Make a prediction using the model
def predict(images):
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        # transforms.RandomResizedCrop((110, 110), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=15),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        # RandomCutout(mask_size=10),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    predicted_classes = []
    
    
    model = NeuralNet()
    model.load_state_dict(torch.load('trained_net.pth'))  # Load the saved weights
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculation
        for image_path in images:
            # Load the image and apply transformations
            img = Image.open(image_path).convert('RGB')
            # transform function, often a set of operations like resizing, normalizing, or converting the image to a tensor.
            img = transform(img)
            #unsqueeze(0) adds a batch dimension to the tensor. PyTorch models typically expect input in the shape of (batch_size, channels, height, width). Here, img is likely a tensor with shape (C, H, W), and unsqueeze(0) changes its shape to (1, C, H, W), where 1 represents a batch size of 1.
            img = img.unsqueeze(0)  # Add batch dimension (1, C, H, W)

            # Move the image to the device used by the model - This moves the image tensor to the same device (CPU or GPU) as the model. next(model.parameters()).device fetches the device type (e.g., cuda:0 for the first GPU) from the model’s parameters, ensuring that the input tensor is compatible with the model during inference or training
            img = img.to(next(model.parameters()).device)
            output = model(img)  # Forward pass
            _, predicted_class = torch.max(output, 1)  # Get the predicted class
            predicted_classes.append(predicted_class.item())

    return predicted_classes  # Return list of predicted class indices

def scan_image(data):
    # Perform image scan logic here
    # `data` could be a list of pixel values or any other image-related data
    result = f"Scanned image with data: {data}"
    return result

def delete_temp_images():
    print(f"Deleeting images after scan complete")

# Open an image file PIL array
def get_images_cropped(image_path):
    try:
        save_dir = 'temp'
        # Open the image
        img = Image.open(image_path)
        
        # Convert the image to RGB (if it's not already)
        img = img.convert("RGB")

        # Resize the image to 880x880
        img = img.resize((1792, 1792))
        
        # Get the width and height of the image
        width, height = img.size
        
        # Ensure the image is evenly divisible into 8x8 squares
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Image dimensions must be divisible by 8")

        # Calculate the width and height of each square
        square_width = width // 8
        square_height = height // 8

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # List to store pixel data for each square (tensor)
        squares = []

        # Loop through each square (row and column)
        # This is croping an image 
        for col in range(0, width, square_width):
            for row in range(0, height, square_height):
                box = (col, row, col + square_width, row + square_height)
                cropped_square = img.crop(box)
                file_path = os.path.join(save_dir, f'micro_{col}_{row}.jpg')
                cropped_square.save(file_path)

                # Append the file path to squares list
                squares.append(file_path)
               
        return squares

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []

# Open an image file and convert it to a NumPy array
def get_image_pixels_numpy(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)  # Convert to a NumPy array
        return img_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def scan_pieces(file):
    # Using the factory to create a piece
    piece = PieceFactory.create_piece("rook")
    piece.file = file  # This will trigger the setter

    squares_images = get_images_cropped(file)
    # You can call this to predict the class based on the processed image
    predicted_classes = predict(squares_images)  # Make the prediction

    # Print the class name
    class_names = ['Queen-Resized', 'Rook-resize', 'bishop-resized', 'knight-resize', 'pawn-resized']
    # Print the predicted classes for all images
    for index, predicted_class in enumerate(predicted_classes):
        # Ensure the predicted_class index is within the range of class_names
        if predicted_class < len(class_names):
            print(f"Predicted Class for square {index + 1}: {class_names[predicted_class]}")
        else:
            print(f"Predicted Class for square {index + 1}: Unknown class (index {predicted_class})")

    # print(f"Predicted Class: {class_names[predicted_class]}")
    
    # see about assigning more weights to class classification
#     class_counts = [100, 200, 300, 400, 50, 250]  # Example class counts
# class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    
    return squares_images
