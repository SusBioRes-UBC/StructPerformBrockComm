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
import ast
import pandas as pd
import os
import Darts.brock_comm_config as config
from Darts.brock_comm_CLT_perform import Darts_CLT_Perform
from Darts.regressor_helper import RegressHelp
from darts import TimeSeries
from darts.models import ARIMA, RegressionModel, LightGBMModel
import numpy as np


def Darts_Pipeline():
    #helper function 
    def append_to_excel(fpath, df, sheet_name):
        with pd.ExcelWriter(fpath,engine='openpyxl', mode="a", if_sheet_exists='replace') as f:
            df.to_excel(f, sheet_name=sheet_name)

    fileList = os.listdir('TALLWOOD DATA/BCTW Sensor Data')
    #fileList = ["Floor 3.csv", "Floor 4.csv"]
    
    # If you want to save running time by using aggregate data, set agg==True; otherwise iterate the orginal complete dataset using False
    agg=True
    
    MAE_dict = {}
    model_mae_dict = {}
    prediction_dict = {}
    forecasts_all_dict = {}

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
        print(regressor)
        regressor_lst = [(['MEAN_TEMPERATURE','TOTAL_PRECIPITATION'],regressor)]
        cov_series = TimeSeries.from_dataframe(regressor, 'ds',['MEAN_TEMPERATURE', 'TOTAL_PRECIPITATION'], fill_missing_dates=True, freq='H', fillna_value=0)
        ws = trial_1.worksheet
        nameList = list(ws)
        nameList.remove('DateTime')
        nameList.remove('Date')
        nameList.remove('Time')
        #nameList = ['W 3rd Edge MC1A (8912/19)']
        if agg==True:
            nameList = ['Aggregate']
        
        ''''
        gbmparam = {
            'lags': [[i for i in range(-300,0)]],
            'lags_past_covariates': [[i for i in range(-300,0)]],
            'lags_future_covariates': [[i for i in range(-300,0)]],
            'output_chunk_length': [300]
        }'''

        modelList = {
            "ARIMA": ARIMA(12,0,0),
            "RegressionModel": RegressionModel(None, None, [i for i in range(-(forecast_horizon-1),1)]),
            "LightGBMModel": LightGBMModel(None, None, [i for i in range(-(forecast_horizon-1),1)])
            #"RegressionModel": RegressionModel(None, None, [i for i in range(-300,1)]),
            #"LightGBMModel": LightGBMModel([i for i in range(-300,0)], [i for i in range(-300,0)], [i for i in range(-300,0)], 300)
        }
    
        modelNameList = []
    
        for modelName, Model in modelList.items():
            modelNameList.append(modelName)
            for columnName in nameList:
                # Create prediction results DataFrame
                df = pd.DataFrame(columns=['ds', 'y'])
                trial_1.preprocess(col_name=columnName, in_sample_forecast=True, forecast_horizon=forecast_horizon, impute='mean', regressor_list=regressor_lst)
                t = trial_1.train_df.copy()
                #t.columns = [c for c in t.columns if c not in ['ds']]
                t['ds'] = pd.to_numeric(pd.to_datetime(t['ds']))
                #print(trial_1.test_df['ds'][0])
                '''
                gridmodel = Model.gridsearch(
                    parameters = gbmparam,
                    series =  TimeSeries.from_dataframe(t, fill_missing_dates=True, freq='H', fillna_value=0),
                    past_covariates = cov_series,
                    future_covariates = cov_series,
                    forecast_horizon = forecast_horizon,
                    stride=1, 
                    start=0.1, 
                    last_points_only=False, 
                    #val_series=TimeSeries.from_dataframe(t1), 
                    #use_fitted_values=True, 
                    metric = mean_absolute_error, 
                    reduction=np.mean, 
                    verbose=True, 
                    n_jobs=1, 
                    n_random_samples=None
                )'''
                trial_1.train_forecast_eval(trial_1.train_df, cov_series, forecast_horizon, groundtruth=trial_1.test_df, Name=modelName, model=Model)
                data = trial_1.forecast_results[['ds', 'y']]
                df = df.append(data, ignore_index=True)
                prediction_dict[modelName] = df
    
                # print(df)
                print('This is the ' + modelName + ' forecast in ' + i)
        
    
        #produce MAE df for each file
        # for method, result in trial_1.eval_results_dict.items():
        #     example_dict = dict({i[:-4] + '_aggr': result})
        for eval_results_dict in trial_1.eval_results_dict_list:
            #print(trial_1.eval_results_dict_list)
            category = eval_results_dict.split("_")[0]
            res_dict = ast.literal_eval(eval_results_dict.split("_")[1])
            result = res_dict['MAE']
            example_dict = dict({i[:-4] + '_aggr': result})
            model_mae_dict[( category, i[:-4] + '_aggr' )] = result
    
            try :
                MAE_dict[category].append(example_dict)
                
            except:
                MAE_dict[category] = [example_dict]
        
        # Make Prediction results sheets
        #print(prediction_dict)
        
        for m, dataframe in prediction_dict.items():
    
            #produce prediction results sheet
            if agg: sheet_name_tail = "aggr"
            else :  sheet_name_tail = ""
            append_to_excel('Darts\output.xlsx', dataframe, i[:-4] + "-" + m + "-" + sheet_name_tail)
        
        forecasts_all_dict[i[:-4]] = prediction_dict
        prediction_dict = {}
    
    trial_1.plot_results('MAE', MAE_dict, Name = modelNameList)
    print(forecasts_all_dict)
    
    """
    ================
    Generate MAE sheets
    ================
    """
    m_dict = {}
    
    for m,f in model_mae_dict.keys():
        m_dict[m] = pd.DataFrame(columns=['Floor', 'MAE'])
    
    for (m,f), v in model_mae_dict.items():
        df = pd.DataFrame([[f,v]], columns=['Floor', 'MAE'])
        #print(df)
        tojoin = [m_dict[m], df]
        m_dict[m] = pd.concat(tojoin)
    #print(m_dict)
    
    for m,d in m_dict.items():
        d = d.reset_index(drop=True)
        append_to_excel('Darts\Performance Metric - MAE.xlsx', d, m)

    return MAE_dict, forecasts_all_dict





    

