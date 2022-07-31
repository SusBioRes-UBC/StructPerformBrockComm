"""
This script conducts the time-series analysis for the following target variables for UBC Brock Commons Tallwood House:
    - moisture performance of CLT
    - vertical movement
Author: Leo Sun, Manqin Cai
References:
- StructPerformBrockComm Prophet(folder)
"""
"""
================
Import libraries
================
"""
import pandas as pd
import os
import brock_comm_config as config
from brock_comm_CLT_perform import Darts_CLT_Perform
from regressor_helper import RegressHelp
from darts import TimeSeries
from darts.models import AutoARIMA, ARIMA, RegressionModel, StatsForecastAutoARIMA, LightGBMModel
#helper function 
def append_to_excel(fpath, df, sheet_name):
    with pd.ExcelWriter(fpath, engine='openpyxl', mode="a", if_sheet_exists='replace') as f:
        df.to_excel(f, sheet_name=sheet_name)

fileList = os.listdir('TALLWOOD DATA/BCTW Sensor Data')
# fileList = ['Floor 3.csv']   

# If you want to save time by using aggregate data, you can set agg=True; If you want to iterate original dataset, use False
agg=True

modelList = {
        "AutoARIMA": AutoARIMA(),
        "ARIMA": ARIMA(12,0,0),       
        "RegressionModel": RegressionModel(None, None, [i for i in range(-299,1)]),
        "LightGBMModel": LightGBMModel(None, None, [i for i in range(-299,1)])
    }

for modelName, Model in modelList.items():
    # create MAE dataframe
    MAE_df = pd.DataFrame()

    for i in fileList:
        # prepare variables
        wb_name = i # sys path will be added within Darts_CLT_Perform
        forecast_horizon = 300 # = 600 hr (interval is 2hr)
        forecast_params = {
            'periods': forecast_horizon, 
            'freq': '2H',
        }
        
        trial_1 = Darts_CLT_Perform(wb_name, agg)
        
        climate_data_csv = os.path.sep.join([config.CLIMATE_DATA_PATH,'Haney_UBC_RF_ADMIN_climate_daily_2016-2020.csv'])
        regress_try = RegressHelp()
        regressor = regress_try.prepare_climate_regr(climate_data_csv, convert_day_to_hour_interval='2H',impute='mean')
        regressor_lst = [(['MEAN_TEMPERATURE','TOTAL_PRECIPITATION'], regressor)]
        cov_series = TimeSeries.from_dataframe(regressor, 'ds',['MEAN_TEMPERATURE', 'TOTAL_PRECIPITATION'], fill_missing_dates=True, freq='H', fillna_value=0)
        ws = trial_1.worksheet
        nameList = list(ws)
        nameList.remove('DateTime')
        nameList.remove('Date')
        nameList.remove('Time')
        #nameList = ['W 3rd Edge MC1A (8912/19)']
        if agg==True:
            nameList = ['Aggregate']
        
        # Create prediction results DataFrame for one model one floor
        df = pd.DataFrame(columns=['ds', 'y'])
    
        for columnName in nameList:
            trial_1.preprocess(col_name=columnName, in_sample_forecast=True, forecast_horizon=forecast_horizon, impute='mean', regressor_list=regressor_lst)
            trial_1.train_forecast_eval(trial_1.train_df, cov_series, forecast_horizon, groundtruth=trial_1.test_df, Name=modelName, model=Model)
        
            # initialize list of lists
            data = trial_1.forecast_results[['ds', 'y']]
            df = df.append(data, ignore_index=True)
            #print(df)
            #trial_1.plot_results('in-sample forecast results_with regr', trial_1.forecast_results, Name=modelName)
            print('This is the ' + modelName + ' forecast in ' + i)    
    
        #produce MAE df for each file
        for method, result in trial_1.eval_results_dict.items():
            example_dict = dict({i[:-4] + '_aggr': result})
        df_dictionary = pd.DataFrame.from_dict(example_dict,orient='index')
        df_dictionary = df_dictionary.loc[:,~df_dictionary.columns.duplicated()].reset_index()
        MAE_df = MAE_df.loc[:,~MAE_df.columns.duplicated()].reset_index(drop=True).append(df_dictionary)
        #print(MAE_df)
        
        #produce prediction results sheet
        if agg: sheet_name_tail = "agg"
        else :  sheet_name_tail = ""
        append_to_excel('Darts\output.xlsx', df, i[:-4] + "-" + modelName + "-" + sheet_name_tail)
    
    #make MAE sheet
    MAE_df.rename(columns = {0:'MAE'}, inplace = True)
    append_to_excel("Darts\Performance Metric - MAE.xlsx", MAE_df, 'MAE '+ modelName)


                


    




