# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset into a pandas DataFrame
df = pd.read_csv('variant1.csv')

# Label encode categorical variables
le = LabelEncoder()
df['waterfront'] = le.fit_transform(df['waterfront'])
df['view'] = le.fit_transform(df['view'])
df['condition'] = le.fit_transform(df['condition'])
df['grade'] = le.fit_transform(df['grade'])

# Split the dataset into training, validation, and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

# Define the features and target variable
features = ['bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated']
target = 'price'


# Linear Regression Model
lr = LinearRegression()

# Train the model on the training data
lr.fit(train[features], train[target])

# Use the validation set to tune hyperparameters
lr_pred_val = lr.predict(val[features])
lr_mse_val = mean_squared_error(val[target], lr_pred_val)
lr_r2_val = r2_score(val[target], lr_pred_val)

# Evaluate the model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared
lr_pred_test = lr.predict(test[features])
lr_mse_test = mean_squared_error(test[target], lr_pred_test)
lr_rmse_test = np.sqrt(lr_mse_test)
lr_r2_test = r2_score(test[target], lr_pred_test)

# Random Forest Regression Model
rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)

# Train the model on the training data
rf.fit(train[features], train[target])

# Use the validation set to tune hyperparameters
rf_pred_val = rf.predict(val[features])
rf_mse_val = mean_squared_error(val[target], rf_pred_val)
rf_r2_val = r2_score(val[target], rf_pred_val)

# Evaluate the model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared
rf_pred_test = rf.predict(test[features])
rf_mse_test = mean_squared_error(test[target], rf_pred_test)
rf_rmse_test = np.sqrt(rf_mse_test)
rf_r2_test = r2_score(test[target], rf_pred_test)

# Define an instance of the SVM model:
svm = SVR(kernel='rbf')

# Train the SVM model on the training data
svm.fit(train[features], train[target])

# Use the validation set to tune hyperparameters
svm_pred_val = svm.predict(val[features])
svm_mse_val = mean_squared_error(val[target], svm_pred_val)
svm_r2_val = r2_score(val[target], svm_pred_val)

# Evaluate the model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared
svm_pred_test = svm.predict(test[features])
svm_mse_test = mean_squared_error(test[target], svm_pred_test)
svm_rmse_test = np.sqrt(svm_mse_test)
svm_r2_test = r2_score(test[target], svm_pred_test)

# Comparison of Metrics
print('Linear Regression Model:')
print('Validation Set Metrics:')
print('MSE:', lr_mse_val)
print('R-squared:', lr_r2_val)
print('Test Set Metrics:')
print('MSE:', lr_mse_test)
print('RMSE:', lr_rmse_test)
print('R-squared:', lr_r2_test)

print('\nRandom Forest Regression Model:')
print('Validation Set Metrics:')
print('MSE:', rf_mse_val)
print('R-squared:', rf_r2_val)
print('Test Set Metrics:')
print('MSE:', rf_mse_test)
print('RMSE:', rf_rmse_test)
print('R-squared:', rf_r2_test)

print('\nSupport Vector Machine Regression Model:')
print('Validation Set Metrics:')
print('MSE:', svm_mse_val)
print('R-squared:', svm_r2_val)
print('Test Set Metrics:')
print('MSE:', svm_mse_test)
print('RMSE:', svm_rmse_test)
print('R-squared:', svm_r2_test)

if lr_mse_test < rf_mse_test and lr_mse_test < svm_mse_test:
    print('\nLinear Regression Model performed better on the test set with lower MSE')
elif rf_mse_test < lr_mse_test and rf_mse_test < svm_mse_test:
    print('\nRandom Forest Regression Model performed better on the test set with lower MSE')
else:
    print('\nSupport Vector Machine Regression Model performed better on the test set with lower MSE')
