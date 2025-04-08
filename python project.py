import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data for appliances
appliances = {
    'Fan': {'watt': 75, 'hours': 4},
    'Light': {'watt': 20, 'hours': 5},
    'TV': {'watt': 120, 'hours': 3},
    'Refrigerator': {'watt': 150, 'hours': 24},
    'Computer': {'watt': 200, 'hours': 2}
}

cost_per_kwh = 0.12  # dollars

# Calculate daily, monthly energy use and cost
results = []
for appliance, data in appliances.items():
    watt = data['watt']
    hours = data['hours']
    kwh_per_day = (watt * hours) / 1000  # kWh
    monthly_kwh = kwh_per_day * 30
    monthly_cost = monthly_kwh * cost_per_kwh
    results.append({
        'Appliance': appliance,
        'Watt': watt,
        'Hours': hours,
        'Monthly_kWh': monthly_kwh,
        'Monthly_Cost': monthly_cost
    })

df = pd.DataFrame(results)
print(df[['Appliance', 'Monthly_kWh', 'Monthly_Cost']])

# --- Pie Chart of Energy Usage ---
plt.figure(figsize=(6,6))
plt.pie(df['Monthly_kWh'], labels=df['Appliance'], autopct='%1.1f%%', startangle=140)
plt.title('Monthly Energy Consumption Share')
plt.show()

# --- Bar Chart of Cost per Appliance ---
plt.figure(figsize=(8,5))
plt.bar(df['Appliance'], df['Monthly_Cost'], color='skyblue')
plt.xlabel('Appliance')
plt.ylabel('Monthly Cost ($)')
plt.title('Monthly Energy Cost per Appliance')
plt.grid(axis='y', linestyle='--')
plt.show()

# --- Prediction of Energy Consumption ---
# Simulate usage growth and predict consumption
usage_hours = np.array([data['hours'] for data in appliances.values()])
monthly_kwh = np.array(df['Monthly_kWh'])

X = usage_hours.reshape(-1, 1)
y = monthly_kwh

model = LinearRegression()
model.fit(X, y)

# Predict future energy if hours increase
future_hours = np.array([1, 3, 5, 7, 10]).reshape(-1, 1)
predicted_kwh = model.predict(future_hours)

# Plot prediction
plt.figure(figsize=(8,5))
plt.plot(future_hours, predicted_kwh, marker='o', color='green')
plt.xlabel('Daily Usage Hours')
plt.ylabel('Predicted Monthly kWh')
plt.title('Predicted Monthly Energy Consumption vs Usage Hours')
plt.grid(True)
plt.show()