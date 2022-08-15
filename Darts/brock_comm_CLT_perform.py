"""
This script conducts the time-series analysis for the following target variables for UBC Brock Commons Tallwood House:
    - moisture performance of CLT
    - vertical movement
Author: Manqin Cai, Leo Sun
References:
- StructPerformBrockComm Prophet(folder)
"""
"""
================
Import libraries
================
"""

from darts import TimeSeries
import pandas as pd
import numpy as np
import Darts.brock_comm_config as config
import os
from datetime import datetime as dt
import logging
from sklearn.impute import SimpleImputer
from Darts.regressor_helper import RegressHelp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
class Darts_CLT_Perform:
    """
    This class conducts the time-series analysis for a SINGLE csv file
    """
    def __init__(self, csv_file_name, agg):
        self.eval_results_dict_list = []
        # load datasheet
        sheet_path = os.path.sep.join([config.DATASHEETS_PATH, csv_file_name])
        self.worksheet = pd.read_csv(sheet_path, index_col=False)
        """
        =================
        set up the logger
        =================
        """
        # gets or creates a logger
        self.logger = logging.getLogger(__name__)  
        # set log level
        self.logger.setLevel(logging.INFO)
        # define file handler and set formatter
        file_handler = logging.FileHandler('CLT_perform.log')
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        # add file handler to logger
        self.logger.addHandler(file_handler)
        """
        =============
        data cleaning
        =============
        """
        # - remove white space for column names
        self.worksheet.columns = [x.strip() for x in list(self.worksheet.columns)]
        # - remove white space for str objects in data columns & replace 'NULL' with 'None'
        for col_name in self.worksheet.columns:
            self.worksheet[col_name] = self.worksheet[col_name].apply(lambda x: x.strip() if isinstance(x, str) else x)
            self.worksheet[col_name] = self.worksheet[col_name].apply(lambda x: np.nan if x == 'NULL' else x)
        
        # adding aggregate data column-------------feel free to modify to customize calculation
        if agg == True:
            self.worksheet['Aggregate'] = self.worksheet.iloc[:,1:].astype(float).mean(axis=1, skipna=True)
            #print(self.worksheet)
        # - create two new columns to store Date and Time separately
        self.worksheet["DateTime"] = self.worksheet["DateTime"].apply(lambda x: dt.strptime(x[:-5], "%Y-%m-%d %H:%M:%S")) # slice to exclude timezone info
        self.worksheet['Date'] = self.worksheet["DateTime"].apply(lambda x: x.date())
        self.worksheet['Time'] = self.worksheet["DateTime"].apply(lambda x: x.time())
        # - get a list of data column names 
        self.data_columns = [col_name for col_name in self.worksheet.columns if col_name not in ['Date', 'Time', 'DateTime']]
    
    def preprocess(self, col_name, in_sample_forecast=True, forecast_horizon=None, **kwargs):
        """
        this method does the following for ONE column of a dataframe:
            - prepare train data (and test data, if in-sample forecast)
            - forecast_horizon: must have the same value as 'periods' in forecast_params
        """
        # store column name for plot use
        self.col_name = col_name
        # get a list of timestamps where there is missing data
        boolean_mask= pd.isna(self.worksheet[col_name])
        missing_data_timestamps = list(self.worksheet['DateTime'][boolean_mask])
        # log the information of the data
        self.logger.info(f"==== statistics of {col_name} column ====")
        #self.logger.info(f"timestamps of missing data are {missing_data_timestamps}")
        self.logger.info(f"idx of the FIRST valid cell is: {self.worksheet[col_name].first_valid_index()}")
        self.logger.info(f"idx of the LAST valid cell is: {self.worksheet[col_name].last_valid_index()}")
        self.logger.info(" ")
        # retain the time series data with valid data
        # determine first and last valid index
        first_valid_idx, last_valid_idx = self.worksheet[col_name].first_valid_index(), self.worksheet[col_name].last_valid_index()
        # make a copy of the part of interest
        self.data_for_anal = pd.DataFrame(self.worksheet[[col_name,"DateTime"]].iloc[first_valid_idx:last_valid_idx+1].copy()).rename(columns={col_name:'y', 'DateTime': 'ds'})
        #self.data_for_anal = pd.DataFrame(self.worksheet[col_name].iloc[first_valid_idx:last_valid_idx+1].copy()).rename(columns={col_name:'y'}) # [caution] .iloc is end-exclusive (while .loc is end-inclusive)
        #print(self.data_for_anal.tail())
        
        # check if impute is intended for ALL data (using sklearn SimpleImputer)
        if 'impute' in kwargs:
            my_imputer = SimpleImputer(strategy=kwargs['impute'])
            y = self.data_for_anal['y'].values
            y = y.reshape(-1, 1)
            y_imputed = my_imputer.fit_transform(y).tolist()
            y_imputed = [y[0] for y in y_imputed]
            #print(f"{y[9,0]} is the same as {y_imputed[9]}")
            self.data_for_anal['y_imputed'] = y_imputed
            self.data_for_anal.drop(columns=['y'], inplace=True)
            self.data_for_anal.rename(columns={'y_imputed': 'y'}, inplace=True)
            print(f"after imputation, there is {self.data_for_anal['y'].isna().sum()} missing pt")
            
        if 'regressor_list' in kwargs:
            reg_help = RegressHelp()
            for (regressor_name_lst, regressor_df) in kwargs['regressor_list']:
                # match the timestep between regressor and time series; regressor_tuple([regressor_col_name1,...,regressor_col_nameN], regressor_dataframe)
                adjusted_regr, self.data_for_anal = reg_help.matching_regr_data(regressor_df, self.data_for_anal)
                for regressor_name in regressor_name_lst:
                    # add regressor data to the timeseries dataframe
                    self.data_for_anal[regressor_name] = adjusted_regr[regressor_name].values
        
        # prepare training and test data
        if in_sample_forecast:
            self.train_df = self.data_for_anal[:-forecast_horizon].copy() # use copy(), otherwise you will get 'SettingWithCopyWarning' when try to add new columns
            self.test_df =  self.data_for_anal[-forecast_horizon:].copy()
        else:
            self.train_df = self.data_for_anal.copy()
        print(self.train_df.tail())
        
    def train_forecast_eval(self, train, covariates, forecast_horizon, **kwargs):
        # print("covariates:", covariates)
        name = kwargs['Name']
        m = kwargs['model']
        series = TimeSeries.from_dataframe(train, 'ds','y', fill_missing_dates=True, freq='H',fillna_value=0)
        # print("series:", series)
        p = series
        if name == "ARIMA": 
            m.fit(series, covariates)
            p = m.predict(forecast_horizon, covariates)
        if name == "RegressionModel" or name == "LightGBMModel": 
            m.fit(series, None, covariates)
            p = m.predict(forecast_horizon, None, None, covariates)
        #print(p)
        self.forecast_results = p.pd_dataframe().rename_axis('ds').reset_index()
        print(self.forecast_results)
        
        if 'groundtruth' in kwargs:
            self.eval_results_dict = self.eval_model(kwargs['groundtruth'], self.forecast_results, forecast_horizon)
            self.eval_results_dict_list.append(name + "_" + str(self.eval_results_dict))
            # log the evaluation results
            for method, result in self.eval_results_dict.items():
                self.logger.info(f"{method}: {result}")                    
        
    def eval_model(self, groundtruth, forecast_results, forecast_horizon):
        """
        Argument:
            - groundtruth: a df with same shape train_df, typically df('ds', 'y')
            - forecast_results: a df with a shape of (n,22), important info ('ds, 'yhat', 'yhat_lower', 'yhat_higher')
        """
        #print(f"groundtruth shape is {groundtruth['y'].shape}")
        #print(f"forecast_results is {forecast_results['yhat'].shape}")
        return {'MAE': mean_absolute_error(groundtruth['y'], forecast_results['y'][-forecast_horizon:])}    
    
    def plot_results(self, fig_name, MAE_dict, **kwargs):
        """
        use the built-in plotting method to plot the forecast results, see: https://unit8co.github.io/darts/generated_api/darts.timeseries.html#darts.timeseries.TimeSeries.plot
        """
        plt.figure(figsize=(10, 8))
        for mae in MAE_dict:               
            MAE_floor_list = []
            MAE_value_list = []
            for m_d in MAE_dict[mae]:
                floor = list(m_d.keys())[0]
                val = list(m_d.values())[0]
                MAE_floor_list.append(floor)
                MAE_value_list.append(val)
            
            plt.plot(MAE_floor_list, MAE_value_list,label=mae)
            
        plt.xlabel("floor")
        plt.ylabel("MAE")
        plt.legend(loc='best')
        plt.savefig(os.path.sep.join([config.OUTPUT_PATH,'{}.png'.format(fig_name)]), dpi=600)
        plt.show()