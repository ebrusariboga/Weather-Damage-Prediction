import joblib
import pandas as pd
import json

def predict(input_file):
    # Load the saved model, scaler, and columns
    best_model = joblib.load('./model/logistic_regression_best_model.pkl')
    scaler = joblib.load('./model/scaler.pkl')
    columns = joblib.load('./model/columns.pkl')

    # Load incoming data from JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Convert the incoming data to a DataFrame
    incoming_df = pd.DataFrame(data['data'])

    # Create Wind Speed Categories
    bins = [0, 5, 15, float('inf')]
    labels = ['Low', 'Medium', 'High']
    incoming_df['WindspeedCategory'] = pd.cut(incoming_df['Windspeed'], bins=bins, labels=labels)

    # One-Hot Encoding for Event and WindspeedCategory
    incoming_encoded_df = pd.get_dummies(incoming_df, columns=['Event', 'WindspeedCategory'])

    # Align columns with the training set
    for col in columns:
        if col not in incoming_encoded_df:
            incoming_encoded_df[col] = 0

    incoming_encoded_df = incoming_encoded_df[columns]

    # Normalize the windspeed feature
    incoming_encoded_df[['Windspeed']] = scaler.transform(incoming_encoded_df[['Windspeed']])

    # Predict probabilities
    probabilities = best_model.predict_proba(incoming_encoded_df)

    # Prepare the response
    response = {
        "data": []
    }
    for i, row in incoming_df.iterrows():
        response["data"].append({
            "id": row["id"],
            "Prob0": round(probabilities[i][0], 2),
            "Prob1": round(probabilities[i][1], 2)
        })

    return response

# Load input data from JSON file and get predictions
input_file = './data/input.json'
predictions = predict(input_file)
print(json.dumps(predictions, indent=4))
