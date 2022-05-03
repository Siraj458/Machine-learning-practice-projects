#import data
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# gather data
df = load_boston()
data = pd.DataFrame(data = df.data, columns=df.feature_names)
data['PRICE'] = df.target

log_price = np.log(data.PRICE)
features = data.drop(['INDUS','PRICE', 'AGE'], axis=1)
target = pd.DataFrame(data=log_price, columns=['PRICE'])

# property estimate, so we need all the mean values of data to make a crude asumption
RM_IDX = 4
PTRATIO_IDX = 8
CHAS_IDX = 2

ZILLOW_INDEX = 583.3  #728.7
DATASET_INDEX = np.median(data.PRICE)
SCALE_FACTOR = ZILLOW_INDEX / DATASET_INDEX

property_stat = features.mean().values.reshape(1,11)

# regression: predicted values, MSE & RMSE
regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features) #yhat
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

# function for log price
def get_log_estimate(nr_rooms,
                    student_per_class,
                    next_to_river=False,
                    high_confidence=True):
    
    # configure property stat
    property_stat[0][RM_IDX] = nr_rooms
    property_stat[0][PTRATIO_IDX] = student_per_class
    
    if next_to_river:
        property_stat[0][CHAS_IDX] = 1
    else:
        property_stat[0][CHAS_IDX] = 0
    
    # Make prediction
    log_estimate = regr.predict(property_stat)[0][0]
    
    #calc range
    if high_confidence:
        upperbound = log_estimate + 2*RMSE
        lowerbound = log_estimate - 2*RMSE
        interval = 95
    else:
        upperbound = log_estimate + RMSE
        lowerbound = log_estimate - RMSE
        interval = 68
    
    return log_estimate, upperbound,  lowerbound, interval

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True, scale=SCALE_FACTOR):
    
    """
    Estimate the property price of boston.
    
    Keyword arguments:
    rm -- number of room.
    ptratio -- student per teacher ratio of school in the area.
    chas -- True if property is near river, False otherwise.
    large_range -- True for 95% prediction interval, False for 68% interval.
    
    """
    
    if rm < 1 or ptratio < 1:
        print('That is unrealistic. Try again!')
        return
        
    
    log_estimate, upperbound, lowerbound, interval =get_log_estimate(rm, ptratio,
                                              next_to_river=chas, high_confidence=large_range)
    
    normal_estimate =np.around(np.e**log_estimate * 1000 * scale, -3)
    upper_bound = np.around(np.e**upperbound * 1000 * scale, -3)
    lower_bound = np.around(np.e**lowerbound * 1000 * scale, -3)
    
    print(f'The estimated property value is {normal_estimate}.')
    print(f'at {interval} confidence the valuation range is')
    print(f'USD {lower_bound} at lower end to USD {upper_bound} at high end.')

    