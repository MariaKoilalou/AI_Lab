# Import necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas DataFrame
df = pd.read_csv('variant1.csv')

# Print the head of the dataset to inspect the values
print(df.head())

# Print the basic information about the dataset
print(df.info())

# Prints descriptive statistics about total count, mean, standard deviation, minimum, maximum, and quartiles
print(df.describe())

# Handling missing data
df = df.dropna()

# Label encode categorical variables
le = LabelEncoder()
df['waterfront'] = le.fit_transform(df['waterfront'])
df['view'] = le.fit_transform(df['view'])
df['condition'] = le.fit_transform(df['condition'])
df['grade'] = le.fit_transform(df['grade'])
df['date'] = pd.to_datetime(df['date'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Scale numeric features
scaler = StandardScaler()
df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
    'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']] = scaler.fit_transform(
    df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']])

# Dataset after scaling
print(df.head())

# Adding the Type column
df['type'] = 'variant1'

# Categorizing quality
df['price_category'] = np.where(df['price'] > 5.401822e+05, 'expensive', 'cheap')

# Encoding categorical variables
df = pd.get_dummies(df, columns=['type', 'price_category'])

features = df.drop(columns=['id', 'date', 'zipcode', 'lat', 'long'])
target = df['price']

print(df['price'])

f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Compute the correlation matrix for the training features
corr_matrix = f_train.corr()

# Sort the correlations with 'price' in descending order
corr_with_price = corr_matrix['price'].sort_values(ascending=False)

# Define the threshold for correlation
corr_threshold = 0.5

# Filter features to only include those with high correlation
high_corr_features = list(corr_with_price[abs(corr_with_price) >= corr_threshold].index)

# Remove the target variable from the list of features
high_corr_features.remove('price')

# Use only high correlation features in the training set
f_train = f_train[high_corr_features]

# Use only high correlation features in the test set
f_test = f_test[high_corr_features]

print(high_corr_features)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(f_train, t_train)
t_pred_lin = lin_reg.predict(f_test)
mse_lin = mean_squared_error(t_test, t_pred_lin)
r2_lin = lin_reg.score(f_test, t_test)
rmse_lin = mean_squared_error(t_test, t_pred_lin, squared=False)
print("Linear Regression modeli:")
print("MSE:", mse_lin)
print("R2 score:", r2_lin)
print("RMSE:", rmse_lin)
