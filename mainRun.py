import joblib 
import pandas as pd
import plotgraph


model = joblib.load('DehydrationModel.pkl')
scaler = joblib.load('scaler.pkl')


def predict_dehydration(input_data):
  
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return 'Yes' if prediction[0] == 1 else 'No'


water_intake = float(input("Enter water intake (liters): "))
physical_activity = float(input("Enter physical activity (hours): "))
ambient_temperature = float(input("Enter ambient temperature (°C): "))
sweat_rate = float(input("Enter sweat rate (liters/hour): "))


input_data = pd.DataFrame({
    'Water_Intake (liters)': [water_intake],
    'Physical_Activity (hours)': [physical_activity],
    'Ambient_Temperature (°C)': [ambient_temperature],
    'Sweat_Rate (liters/hour)': [sweat_rate]
})


result = predict_dehydration(input_data)
print(f"Predicted Dehydration Symptoms: {result}")

# Load the dataset for comparison
data = pd.read_csv('dataset_dehydration.csv')

# Plot comparison graphs
plotgraph.plot_comparison(data, input_data, result)
