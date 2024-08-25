from sklearn.preprocessing  import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd
data = pd.read_csv('dataset_dehydration.csv');
features_to_scale = ['Water_Intake (liters)', 'Physical_Activity (hours)', 'Ambient_Temperature (°C)', 'Sweat_Rate (liters/hour)']
scalar = MinMaxScaler()
data[features_to_scale] = scalar.fit_transform(data[features_to_scale])
data.to_csv('scaled_dataset_dehydration.csv',index=False)
x = data[['Water_Intake (liters)', 'Physical_Activity (hours)', 'Ambient_Temperature (°C)', 'Sweat_Rate (liters/hour)']]
y = data['Dehydration_Symptoms']
y = y.map({'No':0,'Yes':1})
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(scalar, 'scaler.pkl')
y_pred = model.predict(x_test)
joblib.dump(model,'DehydrationModel.pkl')
