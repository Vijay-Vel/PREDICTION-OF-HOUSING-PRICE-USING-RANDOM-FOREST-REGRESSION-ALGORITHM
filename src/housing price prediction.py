import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import random
#colour part for the statical data
r = random.random()
b = random.random()
g = random.random()
color = (r, g, b)
#data set path
file_path="D:\kaggle\dataset.csv"
#read the csv file
home_data = pd. read_csv(file_path)
#print the home data
print("The total data set of the housing price \n \n \n ")
print(home_data)
#print the columns
print("\n\n]The columns of the given data set is listed below \n\n\n\n")
print(home_data.columns)
print(" The statical data for the given data set is shown below :\n\n\n")
home_data.set_index("Id").plot()
plt.grid()
plt.legend()
plt.show()
#drop the unwanted data from source
drop_data= home_data.dropna(axis=0)
#print the drop data
print("The unwanted data which present in the data set is cleared below\n\n\n\n")
print(drop_data)
#select the targeted prediction and assign the value to y(small y)
y=home_data.SalePrice
#print the target
print(" The prediction target is SalesPrice of the house in the Area of 1200 sqft \n ")
print(y)
#select the targeted prediction and assign the value to y(small y)
y=home_data.SalePrice
#print the target
print(" The prediction target is SalesPrice of the house in the Area of 1200 sqft \n ")
print(y)
print(" The statical data for prediction target : \tSALEPRICE\n\n ")
#data set path
fp="D:\kaggle\dataset_y.csv"
#read the csv file
hd_y= pd. read_csv(fp)
hd_y.set_index("Id").plot(c=color)
plt.grid()
plt.legend()
plt.show()
#create the constrains or features and name it as X(capital X)
feature_names = ["LotArea","GrLivArea","KitchenAbvGr","1stFlrSF","2ndFlrSF",
"FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
print(" The necessary features or constrains needed for basic house construction are :")
print(*feature_names)
#assignin the values at X at home data
X=home_data[feature_names]
#print the key constrains
print(X)
#select the model and the specify and fit it .
#from sklearn.tree import DecisionTreeRegressor
#randon tree regressor algorithms
home_model = DecisionTreeRegressor(random_state=0)
#fit the model
home_model.fit(X, y)
#print the model
print(" The home model fitting ")
print(home_model)
#predict the data certain range
predictions = home_model.predict(X)
#print the predictions
print("the precitions are present in the list :",predictions)
#sample prediction values for both target and constrains
print("First in-sample predictions: \n \n ", home_model.predict(X.head()))
print("Actual target values for those homes:\n \n", y.head().tolist())
#model validatation the data and train the split the data
#from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
#specify and fit the model
home_model = DecisionTreeRegressor(random_state=1)
#print the model
#print the model after fit
print("The training of the model progressing in the place :\n\n\n")
home_model.fit(train_X,train_y)
#predict the values
val_predictions=home_model.predict(val_X)
print("The SalesPrice of basically predicted certain abnormality : ")
print(val_predictions)
#check the mean absolut error to verify the data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y,val_predictions)
print("\n\nThe mean absolute error : ",val_mae)
#under fitting and over fitting the model
# find the different types of the leaf nodes annd theri mean absolute error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
  #get random get leaf to find the data
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
#best leaf node example mininmum leaf node is the best
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
print(scores,end=" ")
#best tress size choosing
best_tree_size = min(scores, key=scores.get)
print("\n The best tree size :\t[",best_tree_size,"]")
#fit the model at the final form
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
#generte the final model to prodect the results
final_model.fit(X, y)
#print the final model
print("The final model is about present\n\n ")
print(final_model)
import glob
# path to file you will use for predictions
test_data_path = "D:/kaggle/test.csv"
# read test data file using pandas
test_data = pd.read_csv(test_data_path)
# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X =test_data[feature_names]
# make predictions which we will submit.
test_preds = final_model.predict(test_X)
# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id,
'SalePrice': test_preds})
output.to_csv('D:\kaggle\predict_1.csv',index=False)
print("The Respected prediction has been done and the result will be saved as CSV file formart at the respected location .")
print(" This prediction is done using the Random Regression algorithm")
