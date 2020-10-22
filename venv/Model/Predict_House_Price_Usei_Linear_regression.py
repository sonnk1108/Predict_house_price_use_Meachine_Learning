import pandas as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as num
data = np.read_csv("/media/nguyekhanhson/Study/study/Meachin learning/Data_Set/house_data.csv")
sales = np.DataFrame(data=data)
x = num.array(sales["price"].values).reshape(21613,1)
y=num.array(sales[["sqft_living","bathrooms","bedrooms","floors","sqft_lot","zipcode","condition","grade","waterfront",	"view"	,"condition",	"grade"	,"sqft_above",	"sqft_basement"	,"yr_built"	,"yr_renovated"	,"zipcode"	,"lat",	"long"	,"sqft_living15"	,"sqft_lot15"]].values)
x_train, x_test, y_train, y_test = train_test_split(y, x, test_size=1 / 5, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
house= sales.query('id == 9212900260')
house1=num.array(house[["sqft_living","bathrooms","bedrooms","floors","sqft_lot","zipcode","condition","grade"]].values)

print(regressor.score(x_test,y_test))