# app.py
from flask import Flask, render_template, request
import os
from cd_utils import load_model, predict_image, crop_currency_from_image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            cropped_path = crop_currency_from_image(image_path)
            prediction, confidence = predict_image(cropped_path, model)

    return render_template('index.html', prediction=prediction, confidence=round(confidence, 2) if confidence else None, image=image_path)

if __name__ == '__main__':
    app.run(debug=True)
