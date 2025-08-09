from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from model_utils import load_model, predict_image, CLASS_NAMES
from PIL import Image
import os

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_rice_model_regularized.pth'

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
model = load_model(MODEL_PATH)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Predict
            image = Image.open(filepath).convert('RGB')
            pred_class, confidence = predict_image(model, image)
            return render_template('result.html', filename=filename, pred_class=pred_class, confidence=confidence)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename='uploads/' + filename)

# API endpoint for programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400
    if file and allowed_file(file.filename):
        image = Image.open(file.stream).convert('RGB')
        pred_class, confidence = predict_image(model, image)
        return {'class': pred_class, 'confidence': confidence}
    else:
        return {'error': 'Allowed file types are png, jpg, jpeg'}, 400

if __name__ == '__main__':
    app.run(debug=True)
