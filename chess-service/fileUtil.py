from flask import Blueprint, request, jsonify
# secure_filename is a utility function provided by the werkzeug library (which is part of Flask). It is used to safely handle filenames that are uploaded by users.
from werkzeug.utils import secure_filename
# The os module provides a way to interact with the operating system. It includes functions to handle file paths, directories, and other system-level operations.
import os


routes = Blueprint('routes', __name__)

# Path to save uploaded images temporarily (adjust as needed)
UPLOAD_FOLDER = '/Users/sabina.livny/Desktop/React/chess/chess-service/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to process the image upload and scan it
def process_image_upload(req):
    try:
        if not req:
            raise ValueError("Bad request")

         # Check if 'file' part is in the request
        if 'file' not in req.files:
            raise ValueError("No file part")
        
        file = req.files['file']

        # Check if the file is empty or not selected
        if not file or file.filename == '':
            raise ValueError("No selected file or file is empty.")

        # Check if the file has a valid extension
        if not allowed_file(file.filename):
            raise ValueError("Invalid file format.")

        # Secure the file name and create the full file path
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the uploaded file temporarily
        file.save(file_path)

        # Return the file path for further processing
        return file_path

    except Exception as e:
        # Catch any errors during file handling or scanning
        return ""


def delete_temp_images():
    print("delete_temp_images method called")  # Debug statement
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Folder {UPLOAD_FOLDER} does not exist.")
        return
    
    try:
        for item in os.listdir(UPLOAD_FOLDER):
            item_path = os.path.join(UPLOAD_FOLDER, item)
            
            if os.path.isfile(item_path):
                os.unlink(item_path)  # Remove the image file
                print(f"Deleted image: {item_path}")
                
    except Exception as e:
        print(f"Failed to delete {item_path}: {e}")
