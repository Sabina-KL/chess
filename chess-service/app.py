# # Navigate to your project folder
# cd your-chess-app/server

# # Create a virtual environment
# python3 -m venv venv

# # Activate the virtual environment
# # On macOS and Linux:
# source venv/bin/activate
# # On Windows:
# venv\Scripts\activate

# \Run flask - python app.py   - check in "http://127.0.0.1:5000/." to see that python is running

import os
from flask import Flask, jsonify
from flask_cors import CORS
from routes import routes  # Import the routes blueprint

app = Flask(__name__)
# force debug mode! - switch to true if you want to debug
app.config['DEBUG'] = False
app.config['ENV'] = 'development'

# Ensure the upload folder exists
UPLOAD_FOLDER = '/Users/sabina.livny/Desktop/React/chess/chess-service/uploads'  # Specify your desired folder
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#CORS is used When working with React and Flask on different ports (e.g., React on localhost:3000 and Flask on localhost:5000), you may run into CORS (Cross-Origin Resource Sharing) issues. To solve this, you can install and configure flask-cors
CORS(app)  # Enable CORS for all routes

# Register the blueprint
# In a typical Flask application, app.py initializes the Flask app and sets up configurations, while the blueprint defines routes and views.
# By registering the blueprint using app.register_blueprint(routes), you are telling Flask to include the routes and logic defined in the routes.py file (or whatever you named the blueprint).
app.register_blueprint(routes)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Chess API!"})

#Run this command if you want to change the port: flask run --port 5001
if __name__ == '__main__':
    app.run(debug=True, port=5001) 



# from flask import Flask
# from routes import setup_routes

# app = Flask(__name__)

# # Setup routes
# setup_routes(app)

# if __name__ == '__main__':
#     app.run(debug=True)
