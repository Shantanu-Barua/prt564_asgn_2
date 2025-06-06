#Getting Data and Packages
#----------------
#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats

#load dataset
#Ensure that you open the folder where the databases are situated in
#the visual studio code explorer
full_df = pd.read_excel("AvianDataset.xlsx", sheet_name=['Number', 'Proportion'])

number_df = full_df['Number'] #number spread sheet in xlsx file
proportion_df = full_df['Proportion'] #proportion spreadsheet in xlsx file

#Preprocessing
#-------------
#Seasons Column for Exploratory analysis (used in the boxplot)
number_df['Seasons'] = number_df['Month'] % 12 // 3 + 1

#define x and y values
x_num = number_df.iloc[:,:-1].values #all columns asside from the target (last column)
x_prop = proportion_df.iloc[:,:-1].values #all columns asside from the target (last column)
y = number_df.iloc[:,-1].values #Onlt the target variable (Last column)

#Training Data
#--------------
# Split train and test data (80% training, 20% testing)
x_train_prop, x_test_prop, y_train_prop, y_test_prop = train_test_split(x_prop, y, test_size=0.2, random_state=42)
x_train_num, x_test_num, y_train_num, y_test_num = train_test_split(x_num, y, test_size=0.2, random_state=42)

# Handle missing values for training data
nan_filler = SimpleImputer(strategy='mean')
x_train_prop = nan_filler.fit_transform(x_train_prop)
x_test_prop = nan_filler.transform(x_test_prop)

x_train_num = nan_filler.fit_transform(x_train_num)
x_test_num = nan_filler.transform(x_test_num)

# Standardize features
scale = StandardScaler()
x_train_prop = scale.fit_transform(x_train_prop)
x_test_prop = scale.transform(x_test_prop)

x_train_num = scale.fit_transform(x_train_num)
x_test_num = scale.transform(x_test_num)


#Ridge regression - Proportion sheet
#-------------------------------------
#creating ridge mondel for the proportions sheet in the xlsx file
ridge_model_prop = Ridge()
ridge_model_prop.fit(x_train_prop, y_train_prop)
y_predict_prop = ridge_model_prop.predict(x_test_prop)

#Evaluate model for proportion sheet
r2_prop = r2_score(y_test_prop, y_predict_prop)
mse_prop = mean_squared_error(y_test_prop, y_predict_prop)
mae_prop = mean_absolute_error(y_test_prop, y_predict_prop)

#print proprtion sheet model performance metrics
print("---- Ridge Regression Model: Proportion ----")
print("R^2 score: ", r2_prop)
print("MSE: ", mse_prop) 
print("MAE: ", mae_prop)


#Ridge regression - Number sheet
#--------------------------------
#creating ridge model for the number sheet in the xlsx file
ridge_model_num = Ridge()
ridge_model_num.fit(x_train_num, y_train_num)
y_predict_num = ridge_model_num.predict(x_test_num)

#Evaluate model performance for number sheet
r2_num = r2_score(y_test_num, y_predict_num)
mse_num = mean_squared_error(y_test_num, y_predict_num)
mae_num = mean_absolute_error(y_test_num, y_predict_num)

#print number sheet model performance metrics
print("---- Ridge Regression Model: Number ----")
print("R^2 score: ", r2_num)
print("MSE: ", mse_num) 
print("MAE: ", mae_num)

#Visuals
#-------
#Predicted vs Actual number --> Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test_num, y=y_predict_num, alpha=0.6, label='Predicted vs. Actual')
#fit regression line for visuals
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test_num, y_predict_num)
x_value_num = np.linspace(min(y_test_num), max(y_test_num), 100)
y_value_num = slope * x_value_num + intercept
plt.plot(x_value_num, y_value_num, color='pink', linestyle='--', label='Regression Line')
#style plot by adding lable and titles
plt.xlabel("Actual Total Deaths")
plt.ylabel("Predicted Toal Deaths")
plt.title("Ridge Regression: Avtual vs Predicted Deaths (Number Sheet)")
plt.legend()
#Display R-square, MAE and MSE on the plot
plt.text(x=min(y_test_num), y=max(y_test_num)*0.85, 
         s=f"R²: {r2_score(y_test_num, y_predict_num)}\nMAE: {mae_num}\nMSE: {mse_num}", 
         bbox=dict(facecolor='white', alpha=0.6))
plt.grid(True)
plt.tight_layout()
plt.show()

#Predicted vs Actual Proportion --> Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test_prop, y=y_predict_prop, alpha=0.6, label='Predicted vs. Actual')
#fit regression line for visuals
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test_prop, y_predict_prop)
x_value_prop = np.linspace(min(y_test_prop), max(y_test_prop), 100)
y_value_prop = slope * x_value_prop + intercept
plt.plot(x_value_prop, y_value_prop, color='green', linestyle='--', label='Regression Line')
#style plot by adding titles and labels
plt.xlabel("Actual Total Deaths")
plt.ylabel("Predicted Toal Deaths")
plt.title("Ridge Regression: Avtual vs Predicted Deaths (Number Sheet)")
plt.legend()
#Display R-square, MAE and MSE on the plot
plt.text(x=min(y_test_prop), y=max(y_test_prop)*0.85, 
         s=f"R²: {r2_score(y_test_prop, y_predict_prop)}\nMAE: {mae_prop}\nMSE: {mse_prop}", 
         bbox=dict(facecolor='white', alpha=0.6))
plt.grid(True)
plt.tight_layout()
plt.show()

#Risidual plot num --> diagnose patterns and errors (scattered randomly means unbiased)
residuals = y_test_prop - y_predict_prop
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_predict_prop, y=residuals, alpha=0.7)
plt.axhline(0, color='green', linestyle='--')  # Reference line at 0
plt.title("Ridge Regression Residual Plot (Proportion Sheet)")
plt.xlabel("Predicted Total Deaths")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

#Risidual plot num --> diagnose patterns and errors (scattered randomly means unbiased)
residuals = y_test_num - y_predict_num
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_predict_num, y=residuals, alpha=0.7)
plt.axhline(0, color='pink', linestyle='--')  # Reference line at 0
plt.title("Ridge Regression Residual Plot (Number Sheet)")
plt.xlabel("Predicted Total Deaths")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

#Deaths by season --> box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=number_df, x='Seasons', y='Total death')
plt.title("Seasonal Variation in Deaths")
plt.xlabel("Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)")
plt.ylabel("Deaths")
plt.grid(True)
plt.tight_layout()
plt.show()
