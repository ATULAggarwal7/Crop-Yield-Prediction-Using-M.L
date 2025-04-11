from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and preprocessor
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from form
        crop = request.form['crop']
        crop_year = int(request.form['crop_year'])
        season = request.form['season']
        state = request.form['state']
        area = float(request.form['area'])
        production = float(request.form['production'])
        annual_rainfall = float(request.form['annual_rainfall'])
        fertilizer = float(request.form['fertilizer'])
        pesticide = float(request.form['pesticide'])

        # Create a DataFrame for the custom input
        custom_input = pd.DataFrame({
            'Crop': [crop],
            'Crop_Year': [crop_year],
            'Season': [season],
            'State': [state],
            'Area': [area],
            'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        })

        # Preprocess the input and make prediction
        custom_input_transformed = preprocessor.transform(custom_input)
        predicted_yield = model.predict(custom_input_transformed)[0]

        # Return the result to the user
        return render_template('result.html', yield_prediction=f"{predicted_yield:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
