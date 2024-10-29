
from flask import request, Blueprint, jsonify
from werkzeug.utils import secure_filename
import os
from fileUtil import process_image_upload  # Importing your function
# import requests  # For making requests to another API
from imageScanCalculations import scan_pieces
import pprint #beautifier library for printing complex objects


# Create a blueprint for your routes f
# The Blueprint in Flask is a way to organize your application into reusable components or modules. 
# Instead of defining all your routes directly in the main app.py file, you can use Blueprints to group related routes, which makes your application more modular and easier to maintain.
routes = Blueprint('routes', __name__)

# Dummy API route
# @routes.route('/dummy', methods=['GET'])
# def dummy_api():
#     return jsonify({"message": "Hello from the Flask API!", "status": "success"}), 200

#examples:
# @routes.route('/greet', methods=['GET'])
# def greet_user():
#     name = request.args.get('name', 'World')
#     return jsonify({"message": f"Hello, {name}!"})

# # Example POST route
# @routes.route('/echo', methods=['POST'])
# def echo_data():
#     data = request.get_json()  # Get JSON data from request body
#     return jsonify({"echo": data})

# Route that returns a list of items
# @routes.route('/calculate', methods=['GET'])
# def get_items():
#     items = ['chessboard', 'pawn', 'knight', 'bishop', 'queen', 'king']
#     return jsonify({"items": items})

    # Route to scan image
@routes.route('/scan', methods=['POST'])
def scan_image_route():
    try:
        # Get the image data from the request (file)
        if request:
            file_path = process_image_upload(request)
        else:
            result = "Failed to process the image."

        if file_path:
            # Placeholder scan logic
            pixels_array = scan_pieces(file_path)
            # pprint.pprint(pixels_array)

            result = "Scan completed successfully."
            # You can add actual image processing logic here
        else:
            result = "Failed to process the image."
        
        return jsonify({"message": result, "file_path": file_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
