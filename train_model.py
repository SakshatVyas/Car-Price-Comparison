import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('Cleaned_Car_data.csv')

# Selecting the required features
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Convert categorical variables using OneHotEncoder
categorical_features = ['name', 'company', 'fuel_type']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Create the pipeline with preprocessing and Linear Regression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model using pickle
with open('LinearRegressionModel.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete and saved as 'LinearRegressionModel.pkl'.")
