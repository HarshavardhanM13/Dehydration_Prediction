import joblib 
import pandas as pd
import plotgraph

# Load the trained model and scaler
model = joblib.load('DehydrationModel.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict dehydration
def predict_dehydration(input_data):
    # Ensure input_data is a DataFrame with the correct columns
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return 'Yes' if prediction[0] == 1 else 'No'

# User input
water_intake = float(input("Enter water intake (liters): "))
physical_activity = float(input("Enter physical activity (hours): "))
ambient_temperature = float(input("Enter ambient temperature (°C): "))
sweat_rate = float(input("Enter sweat rate (liters/hour): "))

# Create the input DataFrame with matching column names
input_data = pd.DataFrame({
    'Water_Intake (liters)': [water_intake],
    'Physical_Activity (hours)': [physical_activity],
    'Ambient_Temperature (°C)': [ambient_temperature],
    'Sweat_Rate (liters/hour)': [sweat_rate]
})

# Predict dehydration
result = predict_dehydration(input_data)
print(f"Predicted Dehydration Symptoms: {result}")

# Load the dataset for comparison
data = pd.read_csv('dataset_dehydration.csv')

# Plot comparison graphs
plotgraph.plot_comparison(data, input_data, result)
