#---------------------------------
# Preparing packages adn datasets
#---------------------------------

#Import the packages that will be used in the code for building the model
#and creating visualisations
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm

# Load datasets 
#Ensure that you open the folder where the databases are situated in
#the visual studio code explorer
avian_df = pd.read_csv("Avian.csv") 
seasonality_df = pd.read_csv("Seasonality.csv") 
diagnosis_df = pd.read_csv("Diagnosis.csv")


#-----------------
# Preprocessing:
# Makes the data more usuable for the model
#-----------------

# Group by age and region to get total death counts
grouped_avian = avian_df.groupby(["Age", "Region"])['Death'].sum().reset_index()

# Create a continuous numerical variable: year + fractional month
#--------------------------
#allows for a more precise measurement
df_ym = avian_df.drop(["Age", "Region"], axis=1)
df_ym['Year_mon'] = df_ym['Year'] + (df_ym['Month'] - 1) / 12
df_ym = df_ym[[col for col in df_ym.columns if col != 'Death'] + ['Death']]

# Standardise column names and values:
#--------------------------
#some of the column names are different so it is important to create
#a uniform format to make it consistant across different data sets and
#and easier for the  computer to understand
avian_df = avian_df.rename(columns={'Death': 'deaths'})
diagnosis_df = diagnosis_df.rename(columns={'death': 'deaths'})
seasonality_df = seasonality_df.rename(columns={'District': 'Region'})
avian_df['Region'] = avian_df['Region'].str.lower().str.strip()
diagnosis_df['Region'] = diagnosis_df['Region'].str.lower().str.strip()
seasonality_df["Region"] = seasonality_df['Region'].str.lower().str.strip()

# Region mapping based on geographical location and NHS/ UK statistical region alignments:
#--------------------------
# Regions in the avian and diagnosis databases are the same but are different to the 
# Seasonality dataset, the seasonality dataset is based on regions from the website (Enter site name)
# We are grouping the regions in the avian and diagnosis databases by the regions in the seasonality dataset
region_mapping = {
    "east of england": "east anlia",
    "london": "england se & centra south",
    "east midlands": "england e & ne",
    "east of midlands": "england e & ne", 
    "london": "england se & centra south",
    "north west": "england nw & n wales",
    "west midlands": "midlands",
    "scotland": "scotland",
    "south east": "england se & centra south",
    "south west": "south west",
    "wales": "wales",
    "west midlands": "midlands",
    "yorkshire and the humber": "england e & ne"
}

#create a column called 'Region_Group' that is populated by the list above
#based on the coditions above for both diagnosis and avian
avian_df['Region_Group'] = avian_df['Region'].map(region_mapping)
diagnosis_df['Region_Group'] = diagnosis_df['Region'].map(region_mapping)

# Label encoding of region
#--------------------------
# Combine all region group values that are in the three dataframes
#this makes it easier to encode the values
combined_regions = pd.concat([avian_df['Region_Group'], diagnosis_df['Region_Group'], seasonality_df['Region']])
lable_encode = LabelEncoder()

# Fit the label encoder on the combined regions
lable_encode.fit(combined_regions)

# Apply encoder to each dataset
avian_df['Region_Code'] = lable_encode.transform(avian_df["Region_Group"])
diagnosis_df['Region_Code'] = lable_encode.transform(diagnosis_df["Region_Group"])
seasonality_df['Region_Code'] = lable_encode.transform(seasonality_df['Region'])

# Drop original Region columns for cleaner data
# no longer need te region column as we have the 'Region_Group' column and the 
# 'Region_Code' collumn
avian_df.drop('Region', axis=1, inplace=True)
diagnosis_df.drop('Region', axis=1, inplace=True)
seasonality_df.drop('Region', axis=1, inplace=True)


#-------------------
# Merge Datasets
#-------------------

# Use left merge on datasets to preserve any overlapping rows
#merge on similar rows
avian_seasonality_df = pd.merge(avian_df, seasonality_df, on=['Year', 'Month', 'Region_Code'], how='inner')
full_df = pd.merge(avian_seasonality_df, diagnosis_df, on=['Year', 'Month', 'Region_Code'], how='inner')

# Age_y is the age column from the diagnosis database and Age_x is the age column from the avian database
# Drop age_x and keep Age_y then rename age_y to Age for one-hot encoding
if 'Age_x' in full_df.columns and 'Age_y' in full_df.columns:
    full_df = full_df.drop(columns=['Age_x'])
    full_df = full_df.rename(columns = {'Age_y' : 'Age'})

# One-hot encoding for age
if 'Age' in full_df.columns:
    full_df = pd.get_dummies(full_df, columns=['Age'], drop_first=True, dtype=int)


#-----------
# Features
#-----------

# Time Features
#------------------------
#used to look at patterns or relationships that exist across time
#groups data into quarters and then into seasons
full_df['Quarter'] = pd.to_datetime(full_df[['Year', 'Month']].assign(DAY=1)).dt.quarter
full_df['Season'] = full_df['Month'] % 12 // 3 + 1  # Approximate seasonal groupings
#full_df = pd.get_dummies(full_df, columns=['Month'], prefix='Month', drop_first=True)

#sort for lag
#------------------------
#prepares the data for time series operations (lag) and ensure that
# there is a consistant chronological order within each region
full_df.sort_values(by=['Region_Code', 'Year', 'Month'], inplace=True)

#lag and rolling features
#------------------------
#shifts the 'deaths_x' values down by 1 to allow for the previous months
#deahts to become a new column and thus allow those deaths to act as a predictor
#the calcualtes 3 months of rolling averages, looking at the current month and the 2 prior months
# then computes when there is one value avalable and flattens resulting series back into the 
#original data frame this is done to smooth out short-term fluccuations 
# assisting in capturing seasonal trends
full_df['Lag_Deaths'] = full_df.groupby('Region_Code')['deaths_x'].shift(1)
full_df['RollMean_Deaths'] = full_df.groupby('Region_Code')['deaths_x'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)


#-----------------
# Outlier removal
# Remove outliers within the dataset to prevent data from being skewed
#-----------------

#removes outliers from the dataframe using Interquartile range
def remove_outliers(df): #defines a function that takes in the dataframe "df"
    Q1 = df.quantile(0.25) #lower quartile for each columns
    Q3 = df.quantile(0.75) #upper quartile for each function
    IQR = Q3 - Q1 #spread of the middle 50% of the data
    # Remove outliers that are outside the range [Q1 - 1.5 * IQR, Q3+1.5*IQR]
    #identifies outliers in any column of each row by creating a boolean data frame where 'True'
    #means that there is an outlier
    df_filtered = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)] 
    return df_filtered #returns the filtered data fram wtith the outliers removed


#-------------------------------------------------------------
# Interaction features:
# Assist the model with capturing more complex relationships
#reveals more nuanced effects of weather on deaths
#-------------------------------------------------------------

# Creates interactive features for weather
full_df['Temp_Sunshine'] = full_df['Temperature'] * full_df['Sunshine'] #interaction between the heat and sunlight (help with identifying heat waves)
full_df['Rainfall_Sunshine'] = full_df['Rainfall'] * full_df['Sunshine'] #Used to predict stuff like sunshowers

# Creates interaction features for region and weather
#helps with identifying varying climate vulnerabilities
full_df['Region_Temp'] = full_df['Region_Code'] * full_df['Temperature']
full_df['Region_Rain'] = full_df['Region_Code'] * full_df['Rainfall']

# Creates interaction for age groups and weather
# Manually hotencode age_ column by key weather features 
#finds all columns that start with "Age_" which were previously one-hot encoded
# then creates an interaction
#helps model how weather affects different age groups (e.g. older being more sensitive then younger)
age_cols = [col for col in full_df.columns if col.startswith('Age_')]
for col in age_cols:
    full_df[f'{col}_Temp'] = full_df[col] * full_df['Temperature']
    full_df[f'{col}_Rain'] = full_df[col] * full_df['Rainfall']

#------------------------
#drop NA from lag features
#------------------------

full_df = full_df.dropna()


#-----------------
# Target Features
#-----------------

# Set target features
#features that are trying to be predicged
#removes the target as we are not using it to predict an removes and columns that have not been encoded
x = full_df.drop(columns=['deaths_x']).select_dtypes(exclude='object') #all numerical features asside from the target 'y'
y = full_df['deaths_x'] #sets the variable we are trying to predict


#---------------
# Train model
#---------------

# Train-test split the data
#use 20% of the dataset for testing and 80% for training 
# best results are obtained by using 80%-70% for traing and the remaining for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#----------------
# Scale Features
#----------------

#standardised by removing the mean and scaling to unit variance
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train) #fit scaler on training data and transform it
x_test_scaled = scale.transform(x_test) #transform test data no fitting done to prevent data leakage


#-----------------------------------
# Build the Linear Regression Model
#-----------------------------------

#built and trained model using the scaled data
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)
y_predict = linear_model.predict(x_test_scaled) #predict target values using trained model and scaled test data

# Evaluate Linear Regression model performance
print("---- Linear Regression ----")
print("R^2 Score: ", r2_score(y_test, y_predict)) #R-square for explaining proportion of variance
print("MSE: ", mean_squared_error(y_test, y_predict)) #mean square error
print("MAE: ", mean_absolute_error(y_test, y_predict)) #mean absolute error


#------------------------------
# Build Ridge Regression Model
#------------------------------

#built and trained model using the scaled data
ridge_model = Ridge()
ridge_model.fit(x_train_scaled, y_train)
y_predict_ridge = ridge_model.predict(x_test_scaled) #predict target values using ridge model

#Evaluate ridge regression models performance
print("\n---- Ridge Regression ----")
print("R^2 Score: ", r2_score(y_test, y_predict_ridge)) #R-square score
print("MSE: ", mean_squared_error(y_test, y_predict_ridge)) #Mean square error
print("MAE: ", mean_absolute_error(y_test, y_predict_ridge)) #mean absolute error


#----------------
# Visualisations:
#Visualise what the models were trying to predicts
#----------------

#Actual vs. predicted plot (Linear regression) --> scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_predict, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actal deaths")
plt.ylabel("Predicted Deaths")
plt.title("Linear regression Actual vs predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

#Ridge coefficient bar plot --> Horizontal bar
ridge_coef = pd.Series(ridge_model.coef_, index=x.columns)
plt.figure(figsize=(12, 6))
ridge_coef.sort_values().plot(kind="barh", color="purple")
plt.title("Ridge regression coefficents")
plt.xlabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.show()

#------------
# References
#------------
