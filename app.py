from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from table_detector import detect_table_and_cells
from flask_cors import CORS
import json

app = Flask(__name__)
# Configure CORS to accept requests from any origin
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_table', methods=['POST', 'OPTIONS'])
def process_table():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        print("Received request", request.files)  # Debug print
        
        # Check if image was sent in request
        if 'image' not in request.files:
            print("No image in request")  # Debug print
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        print(f"Received file: {file.filename}")  # Debug print
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Get the name list from the request
        name_list = None
        if 'names' in request.form:
            try:
                name_list = json.loads(request.form['names'])
                print(f"Received {len(name_list)} names")  # Debug print
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid names format'}), 400
            
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"Saved file to: {filepath}")  # Debug print
        
        # Process the image with the name list
        detect_table_and_cells(filepath, name_list)
        print("Image processed successfully")  # Debug print
        
        # Read and return only the results from result.json
        with open('result.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # Clean up - remove the uploaded file
        os.remove(filepath)
        print("File cleanup completed")  # Debug print
            
        return jsonify(results)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server on port 5000...")  # Debug print
    app.run(debug=True, port=5000, host='0.0.0.0') 