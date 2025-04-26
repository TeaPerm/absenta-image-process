from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from table_detector import detect_table_and_cells
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_table', methods=['POST', 'OPTIONS'])
def process_table():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        print("Received request", request.files)  # Debug print
        
        if 'image' not in request.files:
            print("No image in request")  # Debug print
            return jsonify({'message': 'Nincs képfájl megadva'}), 400
        
        file = request.files['image']
        print(f"Received file: {file.filename}")  # Debug print
        
        if file.filename == '':
            return jsonify({'message': 'Nincs fájl kiválasztva'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'message': 'Nem támogatott fájltípus'}), 400
            
        name_list = None
        if 'names' in request.form:
            try:
                name_list = json.loads(request.form['names'])
                print(f"Received {len(name_list)} names")  # Debug print
            except json.JSONDecodeError:
                return jsonify({'message': 'Érvénytelen névlista formátum'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"Saved file to: {filepath}")  # Debug print
        
        # Call the detect_table_and_cells function and get both image and results
        _, results = detect_table_and_cells(filepath, name_list)
        print("Image processed successfully")  # Debug print
        
        # Check if there's an error in the results
        if results and 'message' in results:
            print(f"Error in processing: {results['message']}")
            os.remove(filepath)  # Clean up the uploaded file
            return jsonify(results), 400  # Return the error with 400 Bad Request status
            
        # If no error in results, use the saved JSON file
        with open('result.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        os.remove(filepath)
        print("File cleanup completed")  # Debug print
            
        return jsonify(results)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug print
        return jsonify({'message': f'Hiba történt: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting server on port 5000...")  # Debug print
    app.run(debug=True, port=5000, host='0.0.0.0') 