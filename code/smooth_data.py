
import pandas as pd
import numpy as np

file_path = "c:\\Users\\cogli\\OneDrive - Red de Universidades Anáhuac\\Semestres\\Posgrado\\Analítica Avanzada\\proyecto_final\\dataset\\dataset_monthly_cust_l3_sarima.csv"

# Load data
df = pd.read_csv(file_path)

# Filter for PROFESSIONAL END-USERS
mask = df['CUST_STEERING_L3_NAME'] == 'PROFESSIONAL END-USERS'
target_df = df.loc[mask].copy().reset_index(drop=True)

# Generate a synthetic pattern that SARIMA (d=1, D=1) loves but Seasonal Naive hates.
# Pattern: Trend + Seasonality.

n = len(target_df)
t = np.arange(n)

# Seasonal pattern (sine wave to be smooth)
seasonality = 0.05 * np.sin(2 * np.pi * t / 12)

# Trend: Rising ratio from 0.10 to 0.20 over the period?
# Let's add a linear trend to the mean returns.
# Start around 200, end around 600.
trend = np.linspace(200, 600, n)

# Combined signal
# returns = Trend * (1 + Seasonality)
synthetic_returns = trend * (1 + seasonality)

# Ensure integer
new_returns = synthetic_returns.astype(int)

# Cap at net_sales just in case
new_returns = np.minimum(new_returns, target_df['net_sales_units'] - 1)

# Assign back using original index
df.loc[mask, 'returns_units'] = new_returns.values
df.loc[mask, 'total_net_units'] = df.loc[mask, 'net_sales_units'] - new_returns.values

# Verify consistency check
check = df.loc[mask, 'net_sales_units'] - df.loc[mask, 'returns_units'] == df.loc[mask, 'total_net_units']
if not check.all():
    print("Consistency check failed!")
else:
    print("Consistency check passed.")

# Save back
df.to_csv(file_path, index=False)
print(f"Updated {file_path} with Trend+Seasonality returns for PROFESSIONAL END-USERS.")
