from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

# Load the dataset and model
data = pd.read_csv('Animal_Disease_dataset.csv')
model = pickle.load(open('Random1.pkl', 'rb'))

@app.route('/')
def home():
    AnimalName = sorted(data['AnimalName'].unique())
    symptoms = sorted(
        pd.concat([
            data['symptoms1'],
            data['symptoms2'],
            data['symptoms3'],
            data['symptoms4'],
            data['symptoms5']
        ]).unique()
    )
    return render_template(
        'Home.html',
        AnimalName=AnimalName,
        symptoms=symptoms,
        prediction_text=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    AnimalName = request.form.get('AnimalName')
    symptoms = request.form.getlist('symptoms[]')

    # Ensure there are 5 symptoms (fill with 'none' if fewer)
    while len(symptoms) < 5:
        symptoms.append('none')

    # Create input DataFrame
    input_df = pd.DataFrame([[AnimalName] + symptoms], columns=[
        'AnimalName', 'symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5'
    ])

    # Predict
    prediction = model.predict(input_df)[0]

    # Reload dropdown options
    AnimalName_list = sorted(data['AnimalName'].unique())
    symptoms_list = sorted(
        pd.concat([
            data['symptoms1'],
            data['symptoms2'],
            data['symptoms3'],
            data['symptoms4'],
            data['symptoms5']
        ]).unique()
    )

    return render_template(
        'Home.html',
        AnimalName=AnimalName_list,
        symptoms=symptoms_list,
        prediction_text=f"Predicted Animal Disease: {prediction}"
    )

if __name__ == "__main__":
    app.run(debug=True)
