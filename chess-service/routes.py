
from flask import request, Blueprint, jsonify
from werkzeug.utils import secure_filename
import os
from fileUtil import process_image_upload  # Importing your function
# import requests  # For making requests to another API


# Create a blueprint for your routes
# The Blueprint in Flask is a way to organize your application into reusable components or modules. 
# Instead of defining all your routes directly in the main app.py file, you can use Blueprints to group related routes, which makes your application more modular and easier to maintain.
routes = Blueprint('routes', __name__)

# Dummy API route
@routes.route('/dummy', methods=['GET'])
def dummy_api():
    return jsonify({"message": "Hello from the Flask API!", "status": "success"}), 200

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
    # Get the image data from the request (JSON body)
    data = request.get_json()

    # Call the scan_image function and pass the data
    result = process_image_upload(data, "uploads/")

    # Return the result of the image scan
    return jsonify({"message": result})




# # Route to get image details by ID
# @routes.route('/image/<int:image_id>', methods=['GET'])
# def get_image(image_id):
#     # Call the get_image_details function and pass the image_id
#     image_details = get_image_details(image_id)
#     scan_and_create_pieces('rook', 'white')
#     # Return the image details as JSON
#     return jsonify(image_details)