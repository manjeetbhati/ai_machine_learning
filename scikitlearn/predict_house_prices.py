import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Dataset need to fetched to make this algo work.
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)


predictors = [ 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[predictors]
Y = data.SalePrice

def model_1():
    mod = DecisionTreeRegressor()
    mod.fit(X, Y)
    print("actual Price")
    #print(data.SalePrice)
    #print("--------------")
    #print ("predictions--------------")
    #print(mod.predict(X))
    predicted_prices = mod.predict(X)
    print(mean_absolute_error(Y, predicted_prices))

def model_2():
    train_x, val_x, train_y, val_y = train_test_split(X, Y, random_state=0)
    mod = DecisionTreeRegressor()
    mod.fit(train_x, train_y)
    val_predict = mod.predict(val_x)
    #print(val_x)
    print(mean_absolute_error(val_y, val_predict)) 

model_1()
model_2()
