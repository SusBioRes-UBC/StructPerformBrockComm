import pandas as pd
import os
import Prophet.brock_comm_config as config
from  Prophet.brock_comm_CLT_perform import CLT_perform
from  Prophet.regressor_helper import RegressHelp

def Prophet_Pipeline():
    fileList = os.listdir('TALLWOOD DATA/BCTW Sensor Data')
    #fileList = ["Floor 3.csv", "Floor 4.csv"]
    
    def append_to_excel(fpath, df, sheet_name):
        with pd.ExcelWriter(fpath,engine='openpyxl', mode="a", if_sheet_exists='replace') as f:
            df.to_excel(f, sheet_name=sheet_name)
    
    MAE_df = pd.DataFrame()     # create MAE dataframe
    forecast_dict = {}
    groundtruth_dict = {}
    
    # If you want to save time by using aggregate data, you can let agg==True; If you want to iterate original dataset, use False
    agg=True
    
    for i in fileList:
        # prepare variables
        wb_name = i # sys path will be added within CLT_perform
        forecast_horizon = 300 # = 600 hr (interval is 2hr)
        forecast_params = {
            'periods': forecast_horizon, 
            'freq': '2H',
        }
        
        trial_1 = CLT_perform(wb_name, agg)
    
        climate_data_csv = os.path.sep.join([config.CLIMATE_DATA_PATH,'Haney_UBC_RF_ADMIN_climate_daily_2016-2020.csv'])
        regress_try = RegressHelp()
        regressor = regress_try.prepare_climate_regr(climate_data_csv, convert_day_to_hour_interval='2H',impute='mean')
        regressor_lst = [(['MEAN_TEMPERATURE','TOTAL_PRECIPITATION'],regressor)]
    
        ws = trial_1.worksheet
        nameList = list(ws)
        nameList.remove('DateTime')
        nameList.remove('Date')
        nameList.remove('Time')
        #nameList = ['W 3rd Edge MC1A (8912/19)']
        if agg==True:
            nameList = ['Aggregate']
    
        # Create prediction results DataFrame
        df = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
    
        for columnName in nameList:
            trial_1.preprocess(col_name=columnName, forecast_horizon=forecast_horizon, impute='mean', regressor_list=regressor_lst)
            regressor_trans_func = {
                'MEAN_TEMPERATURE': lambda x: x,
                'TOTAL_PRECIPITATION': lambda x: x,
            }
            trial_1.train_N_forecast(trial_1.train_df,forecast_params,regressor_list=regressor_lst,regr_future=trial_1.test_df, groundtruth=trial_1.test_df)
           
            # initialize list of lists
            data = trial_1.forecast_results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
            df = df.append(data, ignore_index=True)
            print(df)
    
            trial_1.plot_results(i[:-4] +' in-sample forecast results_with regr', trial_1.trained_model, trial_1.forecast_results)
            print('This is the forecast in ' + i)
    
        #produce MAE sheet
        for method, result in trial_1.eval_results_dict.items():
            example_dict = dict({i[:-4] + '_aggr': result})
        df_dictionary = pd.DataFrame.from_dict(example_dict,orient='index')
        df_dictionary = df_dictionary.loc[:,~df_dictionary.columns.duplicated()].reset_index()
        MAE_df = MAE_df.loc[:,~MAE_df.columns.duplicated()].reset_index(drop=True).append(df_dictionary)
        #print(MAE_df)
        
        #add forecast df to dict
        forecast_dict[i[:-4]] = df.rename(columns={'yhat': 'y'})
        
        #produce prediction sheet
        append_to_excel('Prophet\output.xlsx', df, i[:-4] + '_aggr')

        groundtruth_dict[i[:-4]] = trial_1.test_df
    
    MAE_df.rename(columns = {0:'MAE'}, inplace = True)
    append_to_excel('Prophet\Performance Metric - MAE.xlsx', MAE_df, 'Prophet MAE')
    return MAE_df, forecast_dict, groundtruth_dict

