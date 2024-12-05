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
    model.load_state_dict(torch.load('trained_net.pth'))  # Load the saved weights - checkpoint or saved model file created during or after training a neural network. It stores the essential components needed to reuse or deploy the trained model without having to retrain it from scratch.
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculation - gradients are like the "notes" the CNN model creates durring learning phase - we need to remove those and solve efficently without the "learning notes"
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
            
            # Append both file name and predicted class as a tuple or dictionary
            predicted_classes.append({
                'file_name': image_path,
                'predicted_class': predicted_class.item()
            })

    return predicted_classes  # Return list of predicted class indices

# TODO: implement images delete after scaN OR set them as temporary files
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

                file_path = os.path.join(save_dir, f'micro_{col}_{row}.jpg')
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
    predicted_pieces = predict(squares_images)  # Make the prediction

    # Print the class name
    class_names = get_classes() #['queen', 'rook', 'bishop', 'knight', 'pawn', 'king', 'empty']
    
    # Initialize an empty dictionary to store the results
    predictions = {}
    
     # Populate predictions dictionary with file names and predicted classes
    for index, predicted_piece in enumerate(predicted_pieces):
        # Check if the predicted class index is valid within class_names
        if predicted_piece['predicted_class'] < len(class_names):
            class_name = class_names[predicted_piece['predicted_class']]
        else:
            class_name = "Unknown Piece"
        
        # Store in dictionary using the square index as the key
        predictions[index + 1] = {
            'file_name': predicted_piece['file_name'],
            'predicted_class': class_name
        }
        
        # Print each prediction for verification
        print(f"Predicted Class for square {index + 1}: {class_name}")
        
        # TODO: delete files after prediction is done
    
    return predictions
