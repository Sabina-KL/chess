
from piece_factory import PieceFactory
import numpy as np #Python script that opens an image and retrieves the RGB values of each pixel
from PIL import Image #the Pillow library (PIL) to open an image and access its pixel data

# Resulting pixel data using getpixel() This is the Pillow library structure - PIL array
# [
#     (255, 0, 0),    # Pixel at (0,0)
#     (0, 255, 0),    # Pixel at (1,0)
#     (0, 0, 255),    # Pixel at (2,0)
#     (255, 255, 0),  # Pixel at (0,1)
#     (0, 255, 255),  # Pixel at (1,1)
#     (255, 0, 255),  # Pixel at (2,1)
#     (0, 0, 0),      # Pixel at (0,2)
#     (255, 255, 255),# Pixel at (1,2)
#     (128, 128, 128) # Pixel at (2,2)
# ]

# Resulting pixel data using - NumPy library
# array([
#     [[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],  # Row 0
#     [[255, 255,   0], [  0, 255, 255], [255,   0, 255]],  # Row 1
#     [[  0,   0,   0], [255, 255, 255], [128, 128, 128]]   # Row 2
# ])

def scan_image(data):
    # Perform image scan logic here
    # `data` could be a list of pixel values or any other image-related data
    result = f"Scanned image with data: {data}"
    return result

# Open an image file PIL array
def get_image_pixels(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert the image to RGB (if it's not already)
        img = img.convert("RGB")

        # Resize the image to 880X880
        img = img.resize((880, 880))
        
        # Get the width and height of the image
        width, height = img.size
        
        # Ensure the image is evenly divisible into 8x8 squares
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Image dimensions must be divisible by 8")

        # Calculate the width and height of each square
        square_width = width // 8
        square_height = height // 8

        # List to store pixel data for each square
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
                        square_pixels.append(pixel)

                # Add the current square's pixel data to the squares list
                squares.append(square_pixels)
                
        return squares  # A list containing the pixels of 64 squares

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []
    
# def scan_piece_template_images():
    

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

    pixels_array = get_image_pixels_numpy(file) #get image pixels
    squares_pixels = get_image_pixels(file)
    breakpoint()  # Pauses execution here and opens an interactive debugger
    print(pixels_array) #Pdb will open in the terminal, start typing: "p pixels_array"
    
    piece.image_pixels = squares_pixels
    piece.calculate()
    
    return squares_pixels
    # Output the movement logic of the piece
    print(f"Created rook.")


