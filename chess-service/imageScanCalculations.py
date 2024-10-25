from piece_factory import PieceFactory
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from my_neural_net import NeuralNet  # Import the pre-trained network

# Define the necessary transforms (the same ones used during training)
new_transform = transforms.Compose(
    [
        transforms.Resize((880, 880)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load the trained model
def load_model():
    model = NeuralNet()
    model.load_state_dict(torch.load('trained_net.pth'))  # Load the saved weights
    model.eval()  # Set the model to evaluation mode
    return model

# Load and transform the image into a tensor
def load_image(image_path):
    image = Image.open(image_path)
    image = new_transform(image)  # Apply the transformation
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

# Make a prediction using the model
def predict(image_tensor, model):
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_tensor)  # Forward pass
        _, predicted_class = torch.max(output, 1)  # Get the predicted class
    return predicted_class.item()  # Return the class index

def scan_image(data):
    # Perform image scan logic here
    # `data` could be a list of pixel values or any other image-related data
    result = f"Scanned image with data: {data}"
    return result

# Open an image file PIL array
def get_image_pixels_to_tensor(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert the image to RGB (if it's not already)
        img = img.convert("RGB")

        # Resize the image to 880x880
        img = img.resize((880, 880))
        
        # Get the width and height of the image
        width, height = img.size
        
        # Ensure the image is evenly divisible into 8x8 squares
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Image dimensions must be divisible by 8")

        # Calculate the width and height of each square
        square_width = width // 8
        square_height = height // 8

        # List to store pixel data for each square (tensor)
        squares = []

        # Loop through each square (row and column)
        for row in range(8):
            for col in range(8):
                square_pixels = []
                
                # Get the starting and ending coordinates for the current square
                start_x = col * square_width
                start_y = row * square_height
                end_x = start_x + square_width
                end_y = start_y + square_height

                # Loop through the pixels in the current square
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        pixel = img.getpixel((x, y))  # Get the pixel's RGB values
                        # Normalize the pixel values to [-1, 1]
                        normalized_pixel = [(p / 255.0 - 0.5) / 0.5 for p in pixel]
                        square_pixels.append(normalized_pixel)
                
                # Convert the square pixel data to a torch tensor of shape (square_height, square_width, 3)
                square_tensor = torch.tensor(square_pixels).view(square_height, square_width, 3)
                # Transpose to get shape (3, square_height, square_width)
                square_tensor = square_tensor.permute(2, 0, 1)
                
                # Add the current square's tensor data to the squares list
                squares.append(square_tensor)
                
        # Stack the list of square tensors to create a batch of 64 squares
        squares_tensor = torch.stack(squares)
        return squares_tensor  # A tensor containing the 64 squares, shape (64, 3, square_height, square_width)

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

    squares_tensors = get_image_pixels_to_tensor(file)
    # You can call this to predict the class based on the processed image
    model = load_model()  # Load the pre-trained model
    predicted_class = predict(squares_tensors, model)  # Make the prediction

    # Print the class name
    class_names = ['kw', 'qw', 'rw', 'bw', 'kw', 'pw', 'kb', 'qb', 'rb', 'bb', 'kb', 'pb']
    print(f"Predicted Class: {class_names[predicted_class]}")
    
    piece.image_pixels = squares_tensors
    piece.calculate()
    
    return squares_tensors

def execute():
    # Example usage - replace with your image path or logic
    image_path = 'path_to_your_image_file.png'  # Change this to your image path
    squares_tensors = get_image_pixels_to_tensor(image_path)
    if squares_tensors is not None:
        model = load_model()  # Load the pre-trained model
        predicted_class = predict(squares_tensors, model)  # Make the prediction

        # Print the class name
        class_names = ['kw', 'qw', 'rw', 'bw', 'kw', 'pw', 'kb', 'qb', 'rb', 'bb', 'kb', 'pb']
        print(f"Predicted Class: {class_names[predicted_class]}")

if __name__ == "__main__":
    execute()
