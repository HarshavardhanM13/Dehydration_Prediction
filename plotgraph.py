import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison(dataset, input_data, result):
    plt.figure(figsize=(14, 6))

    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=dataset, x='Water_Intake (liters)', y='Physical_Activity (hours)',
                    hue='Dehydration_Symptoms', palette={'Yes': 'red', 'No': 'blue'}, s=50)
    plt.xlabel('Water Intake (liters)')
    plt.ylabel('Physical Activity (hours)')
    plt.title('Dataset: Water Intake vs Physical Activity')
    plt.legend(title='Dehydration Symptoms')

    
    plt.subplot(1, 2, 2)
    plt.scatter(input_data['Water_Intake (liters)'], input_data['Physical_Activity (hours)'],
                c='green' if result == 'Yes' else 'orange', s=100, label=f'Input Data: {result}')
    plt.xlabel('Water Intake (liters)')
    plt.ylabel('Physical Activity (hours)')
    plt.title(f'Input Data: Water Intake vs Physical Activity')
    plt.legend()

    plt.tight_layout()
    plt.show()