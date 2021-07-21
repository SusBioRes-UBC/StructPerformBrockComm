"""
This script conducts the time-series analysis for the following target variables for UBC Brock Commons Tallwood House:
	- moisture performance of CLT
	- vertical movement

Author: Qingshi

References:
- UBC Brock Commons Structural Performance Report Sept 2020: https://sustain.ubc.ca/sites/default/files/UBC%20Brock%20Commons%20Structural%20Performance%20Report%20Sept%202020.pdf
- CCBST 2017 Moisture Performance and Vertical Movement Monitoring of Pre-Fabricated CLT - Paper No 88 
- https://www.kaggle.com/kmkarakaya/missing-data-and-time-series-prediction-by-prophet
"""

"""
================
Import libraries
================
"""
#import pmdarima as pm
import pandas as pd
import numpy as np
import brock_comm_config as config
import os
from datetime import datetime as dt
import logging
from copy import deepcopy
from fb_prophet_train_forecast import FB_prophet_train_forecast
from sklearn.impute import SimpleImputer

class CLT_perform:
	"""
	This class conducts the time-series analysis for a SINGLE csv file
	"""

	def __init__(self, csv_file_name):
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
		"""
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
		self.data_for_anal = pd.DataFrame(self.worksheet[col_name].iloc[first_valid_idx:last_valid_idx+1].copy()).rename(columns={col_name:'y'}) # [caution] .iloc is end-exclusive (while .loc is end-inclusive)
		#print(self.data_for_anal.tail())

		# prepare training and test data
		if in_sample_forecast:
			self.train_df = self.data_for_anal[:-forecast_horizon].copy() # otherwise you will get 'SettingWithCopyWarning' when try to add new columns
			self.test_df =  self.data_for_anal[-forecast_horizon:].copy()
		else:
			self.train_df = self.data_for_anal.copy()

		# check if impute is intended for training data (using sklearn SimpleImputer)
		if 'impute' in kwargs:
			my_imputer = SimpleImputer(strategy=kwargs['impute'])
			y = self.train_df['y'].values
			y = y.reshape(-1, 1)
			y_imputed = my_imputer.fit_transform(y).tolist()
			y_imputed = [y[0] for y in y_imputed]
			#print(f"{y[9,0]} is the same as {y_imputed[9]}")

			self.train_df['y_imputed'] = y_imputed
			self.train_df.drop(columns=['y'], inplace=True)
			self.train_df.rename(columns={'y_imputed': 'y'}, inplace=True)
			print(f"after imputation, there is {self.train_df['y'].isna().sum()} missing pt")

		#print(self.train_df.tail())


	def train_N_forecast(self,train,forecast_params,forecast_horizon,**kwargs):
		# obtain the forecast results
		self.forecast_obj = FB_prophet_train_forecast()
		self.forecast_results = self.forecast_obj.train_forecast(train,forecast_params,forecast_horizon)

		# evaluate the forecast results only when groundtruth data is given
		if 'groundtruth' in kwargs:
			self.eval_results_dict = self.forecast_obj.eval_model(kwargs['groundtruth'], self.forecast_results)
			# log the evaluation results
			for method, result in self.eval_results_dict.items():
				self.logger.info(f"{method}: {result}")

	def plot_results(self, forecast_results):
		pass


