from flask import Flask, render_template, request, jsonify
from model import run_plagiarism_analysis, preprocess_data, create_vector_database  # Import from your model
import os
import numpy as np  # Import NumPy if you're using it for any operations

app = Flask(__name__)

# Load the dataset and create the vector database (do this once at app startup)
data_path = r"C:\Users\Admin\Desktop\plag_model\metadata.csv"  # Adjust the path if needed
source_data = preprocess_data(data_path, 100)  # Adjust sample size if necessary
vector_database = create_vector_database(source_data)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure your HTML file is named `index.html` in the templates folder

@app.route('/check', methods=['POST'])
def check_plagiarism():
    query_text = request.form['text']
    
    if query_text.strip() == '':
        return jsonify({'error': 'Text cannot be empty!'}), 400

    # Call the plagiarism checking logic from model.py
    plagiarism_result = run_plagiarism_analysis(query_text, vector_database, plagiarism_threshold=0.8)

    # Ensure all values in plagiarism_result are standard Python types
    def convert_to_serializable(obj):
        """Convert NumPy types and other non-serializable types to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        elif isinstance(obj, np.float32):
            return float(obj)  # Convert NumPy float32 to Python float
        elif isinstance(obj, np.bool_):
            return bool(obj)  # Convert NumPy boolean to native Python boolean
        return obj  # Return as is for standard types

    # Convert all items in plagiarism_result to be JSON serializable
    serializable_result = {key: convert_to_serializable(value) for key, value in plagiarism_result.items()}

    # Return the result as JSON (can be parsed by the front-end)
    return jsonify(serializable_result)

if __name__ == '__main__':
    host = '127.0.0.1'  # Host
    port = 5000         # Port
    print(f"App is running at http://{host}:{port}")  # Print the host and port to the terminal
    app.run(debug=True, host=host, port=port)
