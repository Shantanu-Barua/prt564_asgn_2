import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_clm = pd.read_excel("Seasonality Dataset.xlsx", sheet_name="Database Table")

# print(df_clm)

### Compile all weather related columns ###
weather_cols = ['Max  temp', 'Min Temp', 'Days of frost in the air (days 0 or below)', 'Hours of rainfall', "Hours of sunshine"]


### Clean column names (in case of trailing spaces) ###
df_clm.columns = df_clm.columns.str.strip()

### Drop rows with missing values in weather columns ###
df_weather = df_clm.dropna(subset=weather_cols)

### Standardize the weather features ###
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_weather[weather_cols])

### Apply PCA to reduce to 1 component ###
pca = PCA(n_components=1)
weather_index = pca.fit_transform(X_scaled)

### Add the weather index to the DataFrame ###
df_weather['Weather_Index'] = weather_index

summary = df_weather.groupby(['Year', 'Month', 'District'])['Weather_Index'].mean().reset_index()

# print(summary)

### Set Seaborn theme ###
sns.set(style="whitegrid")

### Prepare monthly plot ##
monthly_avg = df_weather.groupby('Month')['Weather_Index'].mean().reset_index()

### Prepare district plot ###
district_avg = df_weather.groupby('District')['Weather_Index'].mean().reset_index()

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

### --- Plot 1: Weather Index by Month --- ###
sns.barplot(
    x='Month', y='Weather_Index', hue='Month', data=monthly_avg,
    ax=axes[0], palette='Blues', legend=False, order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
)
axes[0].set_title('Average Weather Index by Month')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Weather Index')

### --- Plot 2: Weather Index by District --- ###
sns.barplot(
    y='District', x='Weather_Index', hue='District', data=district_avg,
    ax=axes[1], palette='Greens', legend=False
)
axes[1].set_title('Average Weather Index by District')
axes[1].set_xlabel('Weather Index')
axes[1].set_ylabel('District')

plt.tight_layout()
plt.show()