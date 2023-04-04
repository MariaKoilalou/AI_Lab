# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset into a pandas DataFrame
df = pd.read_csv('variant1.csv')

# Convert categorical variables to numerical using one-hot encoding or label encoding 
df = pd.get_dummies(df, columns=['waterfront', 'view', 'condition', 'grade'])

# Split the dataset into training, validation, and test sets
# Splitting the dataset into
# training, validation, and test sets means dividing the available data into three separate subsets to be used for
# different purposes in building a predictive model.
# The training set is used to train the model, meaning the model
# learns the relationships between the input features and the target variable. The validation set is used to tune the
# hyperparameters of the model and prevent overfitting. Finally, the test set is used to evaluate the performance of
# the model on new, unseen data.
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

# Define the features and target variable
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
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
if lr_mse_test < rf_mse_test:
    print('\nLinear Regression Model performed better on the test set with lower MSE')
else:
    print('\nRandom Forest Regression Model performed better on the test set with lower MSE')
