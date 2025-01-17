from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('water_quality.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Determine if the water is safe for consumption
    if prediction[0] == 0:
        prediction_text = "Water is NOT SAFE for Consumption"
    else:
        prediction_text = "Water is SAFE for Consumption"
    
    # Render the result
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run()
