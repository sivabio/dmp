from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model("ann_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
 # Collect input features from the form (new feature names)
        features = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4']),
            float(request.form['feature5']),
            float(request.form['feature6']),
            float(request.form['feature7'])
        ]
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Make the prediction
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction_class = int(prediction_prob >= 0.5)  # Convert to 0 or 1 class

        return render_template('result.html', 
                               prediction_class=prediction_class,
                               prediction_probability=round(prediction_prob, 4))
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
