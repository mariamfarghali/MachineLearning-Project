import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score
data = pd.read_csv("housing.csv")
print(data)
data = data.sample(n=len(data), random_state=1)  
data.dropna(inplace = True)

x = data.drop(['median_house_value'], axis = 1)
y = data['median_house_value']

# print()
pd.get_dummies(data.ocean_proximity)
data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'], axis = 1)
# print(data)
data = data[['longitude','latitude',
'housing_median_age','total_rooms',
'total_bedrooms','population',
'households','median_income',
'<1H OCEAN', 'INLAND',
'ISLAND','NEAR BAY','NEAR OCEAN',
'median_house_value']]
# print(data)

train_data, test_data, val_data = data[:18000], data[18000:19217], data[19215:]
# print(train_data.shape, test_data.shape, val_data.shape)
X_train, y_train = train_data.to_numpy()[:, :-1], train_data.to_numpy()[:, -1]
X_val, y_val = val_data.to_numpy()[:, :-1], val_data.to_numpy()[:, -1]
X_test, y_test = test_data.to_numpy()[:, :-1], test_data.to_numpy()[:, -1]

scaler = StandardScaler().fit(X_train[:, :8])

def preprocessor(X):
    A = np.copy(X)
    A[:, :8] = scaler.transform(A[:, :8])
    return A

X_train, X_val, X_test, = preprocessor(X_train), preprocessor(X_val), preprocessor(X_test)


lm = LinearRegression().fit(X_train, y_train)
#trainig error = 68593.05.. , validation error = 71382.4355 we care about validation so it's overfitting
print("training error Linear:" , mse(lm.predict(X_train), y_train, squared=False),"Validation error Linear:" , mse(lm.predict(X_val), y_val, squared=False))
r2 = lm.score(X_test, y_test)
print("R-squared linear :", r2)

knn = KNeighborsRegressor()

param_grid={'n_neighbors':np.arange(1,21,2),'p':[1,2]}
kf=KFold(n_splits=6,shuffle=True,random_state=42)
knn_cv=GridSearchCV(knn,param_grid,cv=kf)
knn_cv.fit(X_train, y_train)
print("training error knn:" , mse(knn_cv.predict(X_train), y_train, squared=False),"Validation error knn:" , mse(knn_cv.predict(X_val), y_val, squared=False))
r2 = knn_cv.score(X_test, y_test)
print("R-squared knn:", r2)

#____________________________graphs____________________


import matplotlib.pyplot as plt

# k_values = range(1, 21)  # Range of K values to test
# r2_values = []

# for k in k_values:
#     knn = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
#     r2 = knn.score(X_val, y_val)
#     r2_values.append(r2)

# # Plot K-values vs. R-squared for K-Nearest Neighbors

# plt.figure(figsize=(10, 6))
# plt.plot(k_values, r2_values, marker='o', linestyle='-', color='b')
# plt.title('K-Values vs. R-squared for K-Nearest Neighbors')
# plt.xlabel('K-values')
# plt.ylabel('R-squared')
# plt.grid(True)
# plt.show()

# # Scatter plot for predicted vs. true values for Linear Regression

# plt.figure(figsize=(10, 6))
# plt.scatter(lm.predict(X_test), y_test, color='blue', alpha=0.5)
# plt.title('Scatter Plot for Predicted vs. True Values (Linear Regression)')
# plt.xlabel('Predicted Values')
# plt.ylabel('True Values')
# plt.grid(True)
# plt.show()

