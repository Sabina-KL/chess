import random
import numpy as np
from PIL import Image
from itertools import product
import os  # Import the os module
import torch
import torchvision.transforms as transforms
from my_neural_net import NeuralNet, get_classes, SQUARE_WIDTH ,SQUARE_HEIGHT  # Import the pre-trained network

# Make a prediction using the model
def predict(images):
    transform = transforms.Compose([
        transforms.Resize([SQUARE_WIDTH ,SQUARE_HEIGHT]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
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

def delete_temp_images():
    print(f"Deleeting images after scan complete")

# Open an image file PIL array
def get_images_cropped(image_path):
    try:
        save_dir = 'uploads'
        # Open the image
        img = Image.open(image_path)
        
        # Convert the image to RGB (if it's not already)
        img = img.convert("RGB")

        # Resize the image to 880x880
        img = img.resize((SQUARE_WIDTH * 8 ,SQUARE_HEIGHT * 8))
        
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
                random_number = random.randint(1000, 9999)  # You can adjust the range if needed

                file_path = os.path.join(save_dir, f'micro_{col}_{row}_{random_number}.jpg')
                cropped_square.save(file_path)

                # Append the file path to squares list
                squares.append(file_path)
               
        return squares

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []

def scan_pieces(file):
    squares_images = get_images_cropped(file)
    # You can call this to predict the class based on the processed image
    predicted_classes = predict(squares_images)  # Make the prediction

    # Print the class name
    class_names = get_classes() #['queen', 'rook', 'bishop', 'knight', 'pawn', 'king', 'empty']
    # Print the predicted classes for all images
    for index, predicted_class in enumerate(predicted_classes):
        # Ensure the predicted_class index is within the range of class_names
        if predicted_class < len(class_names):
            print(f"Predicted Class for square {index + 1}: {class_names[predicted_class]}")
        else:
            print(f"Predicted Class for square {index + 1}: Unknown class (index {predicted_class})")
            
    return squares_images
