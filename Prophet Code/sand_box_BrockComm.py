"""
This is a sandbox to try out different modules
"""

import pandas as pd
from datetime import datetime as dt
import os
import brock_comm_config as config
from brock_comm_CLT_perform import CLT_perform
from regressor_helper import RegressHelp

# prepare variables
wb_name = "String Pots.csv" # sys path will be added within CLT_perform
forecast_horizon = 300 # = 600 hr (interval is 2hr)
forecast_params = {
	'periods': forecast_horizon, 
	'freq': '2H',
}


"""
===============
test FB Prophet
===============

# initiate the obj
trial_1 = CLT_perform(wb_name)
#datasheet = trial_1.worksheet

# run forecast (in-sample)
# preprocess the data
#trial_1.preprocess(col_name='5-6 Floor String Pot (8917/18)', forecast_horizon=forecast_horizon, impute='mean')
#trial_1.train_N_forecast(trial_1.train_df,forecast_params,groundtruth=trial_1.test_df)
#trial_1.plot_results('in-sample forecast results', trial_1.trained_model, trial_1.forecast_results)

# run forecast (out-of-sample)
trial_1.preprocess(col_name='5-6 Floor String Pot (8917/18)', in_sample_forecast=False, forecast_horizon=forecast_horizon, impute='mean')
trial_1.train_N_forecast(trial_1.train_df,forecast_params) # ground truth data can still be provided if any (not in our case tho)
trial_1.plot_results('out-of-sample forecast results', trial_1.trained_model, trial_1.forecast_results)

# run retrain (out-of-sample)
trial_1.preprocess(col_name='4-5 Floor String Pot (8917/17)', in_sample_forecast=False, forecast_horizon=forecast_horizon, impute='mean')
trial_1.train_N_forecast(trial_1.train_df,forecast_params,trained_model='initially_trained_model.json') # ground truth data can still be provided if any (not in our case tho)
trial_1.plot_results('out-of-sample forecast results_RETRAINED', trial_1.trained_model, trial_1.forecast_results)
"""

"""
=========================================
# test prophet with regressor (in-sample)
=========================================
"""
trial_1 = CLT_perform(wb_name)

climate_data_csv = os.path.sep.join([config.CLIMATE_DATA_PATH,'Haney_UBC_RF_ADMIN_climate_daily_2016-2020.csv'])
regress_try = RegressHelp()
regressor = regress_try.prepare_climate_regr(climate_data_csv, convert_day_to_hour_interval='2H',impute='mean')
regressor_lst = [(['MEAN_TEMPERATURE','TOTAL_PRECIPITATION'],regressor)]

trial_1.preprocess(col_name='5-6 Floor String Pot (8917/18)', forecast_horizon=forecast_horizon, impute='mean', regressor_list=regressor_lst)


# need to set up train/test split for climate data, as you can't use a function to create future climate data
# accordingly, ts_data needs to be split too (i.e., you can't run out-of-sample prediction) --> this is already done in CLT_perform.preprocess(in_sample_forecast=True)


#print(regressor.tail(24))
#adjusted_regressor, adjusted_train = regress_try.matching_regr_data(regressor,trial_1.train_df)
#print(adjusted_train.head())
#print(adjusted_regressor.head())
regressor_trans_func = {
	'MEAN_TEMPERATURE': lambda x: x,
	'TOTAL_PRECIPITATION': lambda x: x,
}
trial_1.train_N_forecast(trial_1.train_df,forecast_params,regressor_list=regressor_lst,regr_future=trial_1.test_df, groundtruth=trial_1.test_df)
trial_1.plot_results('in-sample forecast results_with regr', trial_1.trained_model, trial_1.forecast_results)
# test retrain model --> dev on pause
#trial_1.train_N_forecast(trial_1.train_df,forecast_params,trained_model='initially_trained_model.json',regressor_list=regressor_lst,test=trial_1.test_df, groundtruth=trial_1.test_df)
#trial_1.plot_results('in-sample forecast results_with regr_RETRAINED', trial_1.trained_model, trial_1.forecast_results)

