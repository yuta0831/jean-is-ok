
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# 1. Initialize and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 2. Save to a file (usually ends in .pkl or .joblib)
joblib.dump(regressor, "linear_regression_model.pkl")

