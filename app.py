from flask import Flask, render_template, request
import numpy as np
import pickle




# Load the model
with open('smk.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/submit', methods=['POST'])
def submit():
    # Get the numerical inputs from the form
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    TVOC = float(request.form['TVOC'])
    eCO2 = float(request.form['eCO2'])
    Raw_H2 = float(request.form['Raw_H2'])
    Raw_Ethanol = float(request.form['Raw_Ethanol'])
    Pressure = float(request.form['Pressure'])
    PM2_5 = float(request.form['PM2.5'])
    NC0_5 = float(request.form['NC0.5'])
    NC2_5 = float(request.form['NC2.5'])
    CNT = float(request.form['CNT'])
    
    # Create the final features array
    final_features = np.array([[Temperature, Humidity, TVOC,eCO2,Raw_H2,Raw_Ethanol,Pressure,PM2_5 , NC0_5 ,NC2_5, CNT]])





    # Make the prediction
    prediction = model.predict(final_features)[0]

    # Set the prediction text based on the model prediction
    if prediction == 0:
        prediction_text = 'No smoke detected'
    else:
        prediction_text = 'Smoke detected'

    # Render the result template with the prediction text
    return render_template('submit.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)