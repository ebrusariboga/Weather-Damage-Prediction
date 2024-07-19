# WeatherDamagePredictionProject

- This project aims to predict the probability of property damage due to different weather events. 
- The prediction model is built using Logistic Regression with hyperparameter tuning and balanced using SMOTE. 
- The project includes data preprocessing, feature engineering, model training, evaluation, and a script for making predictions on new data. 

## Project Structure:

WeatherDamagePrediction/
│
├── data/
│   └── input.json
│
├── model/
│   ├── logistic_regression_best_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── WeatherDamagePredictionModel.ipynb
└── predict_model.py

## Setup Instructions

### 1.Prerequisites
- Python 3.8
- Required Python packages (listed in requirements.txt)

### 2.Create a Virtual Environment

``` python3.8 -m venv weather_env ``` 

### 3.Activate the Virtual Environment

- On Windows:
``` weather_env\Scripts\activate ```

- On macOS and Linux:
``` source weather_env/bin/activate ``` 

### 4.Install Required Packages

``` pip install -r requirements.txt ``` 

## Running the Project

### 1.Jupyter Notebook

- To explore the model training and evaluation process, open the Jupyter Notebook:

``` jupyter notebook WeatherDamagePredictionModel.ipynb ```

- The notebook includes sections for data encoding, normalization, balancing with SMOTE, model training, prediction, and evaluation.

### 2.Predicting with New Data

- To make predictions with new data, use the predict_model.py script. 
- Ensure the input data file is placed in the data directory.

- Example Command:

``` python predict_model.py ```

## Input Data Format
- The input data should be in a JSON format similar to the following example and saved as data/input.json:

{
    "data": [
        {"id": 5000, "Event": "Tornado", "Windspeed": 16.84},
        {"id": 5001, "Event": "Flood", "Windspeed": 2.7}
    ]
}

## Output
- The script will output the predictions in the console in the following format:

{
    "data": [
        {"id": 5000, "Prob0": 0.38, "Prob1": 0.62},
        {"id": 5001, "Prob0": 0.40, "Prob1": 0.60}
    ]
}

## Detailed Steps

### 1.Data Encoding

- Description: Encode categorical variables using one-hot encoding.
- Implementation: See the Data Encoding section in the WeatherPredictionModel.ipynb notebook.

### 2.Data Normalization

- Description: Normalize the windspeed feature.
- Implementation: See the Data Normalization section in the WeatherDamagePredictionModel.ipynb notebook.

### 3.Data Balancing with SMOTE

- Description: Balance the training data using SMOTE.
- Implementation: See the Data Balancing with SMOTE section in the WeatherDamagePredictionModel.ipynb notebook.

### 4.Model Training and Hyperparameter Tuning

- Description: Train the model using Logistic Regression and tune hyperparameters with GridSearchCV.
- Implementation: See the Model Training and Hyperparameter Tuning section in the WeatherDamagePredictionModel.ipynb notebook.

### 5.Model Prediction

- Description: Make predictions using the trained model.
- Implementation: See the Model Prediction section in the WeatherDamagePredictionModel.ipynb notebook.

### 6.Model Evaluation

- Description: Evaluate the model performance using accuracy, precision, recall, F1 score, and confusion matrix.
- Implementation: See the Model Evaluation section in the WeatherDamagePredictionModel.ipynb notebook.

# Conclusion
- This project demonstrates a comprehensive approach to building and deploying a weather event prediction model. 
- By following the steps outlined in this README, you can easily set up the project on your local machine, explore the Jupyter Notebook, and make predictions using new data.