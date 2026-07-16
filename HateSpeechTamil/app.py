import os
import uuid
from flask import Flask, render_template, request, jsonify

# We import the prediction function directly. 
# This automatically loads PaddleOCR, BLIP, and CLIP models into memory on Flask startup.
from predict import predict_meme

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        # Save file with unique name to prevent filename conflicts
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run prediction pipeline
        result = predict_meme(filepath)
        
        if "error" in result:
            return jsonify({'error': result['error']}), 500
            
        # Include static path URL so client can show the image
        result['image_url'] = f"/static/uploads/{filename}"
        
        return jsonify(result)

if __name__ == '__main__':
    # Clear out any old uploads on startup
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
        except Exception:
            pass
            
    print("\n" + "=" * 50)
    print("  Flask server starting at http://localhost:5000  ")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
