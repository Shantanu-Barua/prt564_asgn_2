#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

#load dataset
full_df = pd.read_excel("AvianDataset.xlsx", sheet_name=['Number', 'Proportion'])

number_df = full_df['Number'] #number spread sheet in xlsx file
proportion_df = full_df['Proportion'] #proportion spreadsheet in xlsx file

#Preprocessing the data
#-----------------------
#drop columns that are highly correlated to avoid data leackage
number_df = number_df.drop(columns=['Adult', 'Immature'])

#Seasons Column for Exploratory analysis
number_df['Seasons'] = number_df['Month'] % 12 // 3 + 1

#

#define x and y values
x_num = number_df.iloc[:,:-1].values
x_prop = proportion_df.iloc[:,:-1].values
y = number_df.iloc[:,-1].values


# Split train and test data
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
ridge_model_prop = Ridge(alpha=1.0)
ridge_model_prop.fit(x_train_prop, y_train_prop)
y_predict_prop = ridge_model_prop.predict(x_test_prop)

#Evaluate model for proportion sheet
r2_prop = r2_score(y_test_prop, y_predict_prop)
mse_prop = mean_squared_error(y_test_prop, y_predict_prop)
mae_prop = mean_absolute_error(y_test_prop, y_predict_prop)

#print proprtion sheet performance
print("---- Ridge Regression Model: Proportion ----")
print("R^2 score: ", r2_prop)
print("MSE: ", mse_prop) 
print("MAE: ", mae_prop)


#Ridge regression - Number sheet
#--------------------------------
#creating ridge model for the number sheet in the xlsx file
ridge_model_num = Ridge(alpha=100.0)
ridge_model_num.fit(x_train_num, y_train_num)
y_predict_num = ridge_model_num.predict(x_test_num)

#Evaluate model performance for number sheet
r2_num = r2_score(y_test_num, y_predict_num)
mse_num = mean_squared_error(y_test_num, y_predict_num)
mae_num = mean_absolute_error(y_test_num, y_predict_num)

#print number sheet performance 
print("---- Ridge Regression Model: Number ----")
print("R^2 score: ", r2_num)
print("MSE: ", mse_num) 
print("MAE: ", mae_num)

#Visuals
#-------
#Predicted vs Actual proportion
plt.figure(figsize=(8, 6))
plt.scatter(y_test_num, y_predict_num, alpha=0.7)
plt.plot([y_test_prop.min(), y_test_prop.max()], [y_test_prop.min(), y_test_prop.max()], 'r--')  # 45-degree line
plt.xlabel("Actual Total Deaths")
plt.ylabel("Predicted Total Deaths")
plt.title("Ridge Regression: Predicted vs Actual")
plt.text(x=min(y_test_prop), y=max(y_test_prop)*0.9, 
         s=f"R²: {r2_score(y_test_prop, y_predict_prop)}\nMAE: {mae_prop}\nMSE: {mse_prop}", 
         bbox=dict(facecolor='white', alpha=0.6))
plt.grid(True)
plt.show()

# #Predicted vs Actual num --> scatter plot


# #Risidual plot num --> diagnose patterns and errors (scattered randomly means unbiased)


#Deaths by season --> box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=number_df, x='Seasons', y='Total death')
plt.title("Seasonal Variation in Deaths")
plt.xlabel("Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)")
plt.ylabel("Deaths")
plt.grid(True)
plt.tight_layout()
plt.show()
