from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from PIL import Image
import io

app = Flask(__name__)

# Load the model and CSV file
model = load_model('CNN_model.keras')  # Ensure this file is in the same directory
df = pd.read_csv('ot.csv')             # Ensure this file is in the same directory

# Class labels (must match your training labels)
class_names = ['bacterial spot', 'early blight', 'late blight', 'leaf mold', 'septoria leaf spot', 'spider mites', 'target spot', 'yellow leaf curl virus', 'healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file selected!", 400
    
    if file:
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((50, 50))
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)
        
        # Get the treatment recommendation
        treatment_row = df[df.iloc[:, 0].str.strip().str.lower() == predicted_class.lower()]
        treatment = treatment_row.iloc[0, 1] if not treatment_row.empty else "No treatment found."
        
        return render_template('result.html', 
                               predicted_class=predicted_class, 
                               confidence=confidence, 
                               treatment=treatment)
    
if __name__ == '__main__':
    app.run(debug=True)