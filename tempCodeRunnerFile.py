from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('heart_diseases.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        input_features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data_as_numpy_array)

        if prediction[0] == 1:
            result = '⚠️ Person has Heart Disease'
        else:
            result = '✅ Person does NOT have Heart Disease'
        
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
