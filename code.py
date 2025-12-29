
# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# read the dataset
dataset=pd.read_csv(r'C:\Users\ADMIN\Downloads\23rd- Poly\23rd- Poly\1.POLYNOMIAL REGRESSION\emp_sal.csv')

# x and y variables
x=dataset.iloc[: , 1:2].values
y = dataset.iloc[:,2].values

# SVR Model
from sklearn.svm import SVR
regressor=SVR(kernel="poly",degree=4,gamma='auto',C=6,epsilon=1.8)
regressor.fit(x,y)


# KNN Model
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=2,algorithm='brute',leaf_size=100,p=1,weights="distance")
knn_reg.fit(x,y)



# Decission Tree 
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='poisson', max_depth=3, random_state=0)
dt_reg.fit(x,y)

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(max_depth=4,criterion= "poisson",random_state=0,n_estimators=6)
rf_reg.fit(x,y)

# predicting a new result with SVR
y_pred_svr =regressor.predict([[6.5]])
y_pred_svr

# predicting a new result with KNN
y_pred_knn = knn_reg.predict([[6.5]])
y_pred_knn

# predicting a new result with Decision Tree
y_pred_dt =dt_reg.predict([[6.5]])
y_pred_dt

# predicting a new result with Random Forest
y_pred_rf =rf_reg.predict([[6.5]])
y_pred_rf

# Visualising the SVR results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the KNN results
plt.scatter(x,y,color='red')
plt.plot(x,knn_reg.predict(x),color='blue')
plt.title('Truth or Bluff (KNN Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Decision Tree results
plt.scatter(x,y,color='red')
plt.plot(x,dt_reg.predict(x),color='blue')
plt.title('Truth or Bluff (Decision Tree Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()  

# Visualising the Random Forest results
plt.scatter(x,y,color='red')
plt.plot(x,rf_reg.predict(x),color='blue')
plt.title('Truth or Bluff (Random Forest Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

import pickle

# Save SVR model
with open('svr_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

# Save KNN model
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_reg, file)

# Save Decision Tree model
with open('dt_model.pkl', 'wb') as file:
    pickle.dump(dt_reg, file)

# Save Random Forest model
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_reg, file)

print("✅ All models saved successfully")