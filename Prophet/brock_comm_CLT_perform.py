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
import Prophet.brock_comm_config as config
import os
from datetime import datetime as dt
import logging
from Prophet.fb_prophet_train_forecast import FB_prophet_train_forecast
from sklearn.impute import SimpleImputer
import json
from prophet.serialize import model_to_json, model_from_json
from Prophet.regressor_helper import RegressHelp


class CLT_perform:
	"""
	This class conducts the time-series analysis for a SINGLE csv file
	"""

	def __init__(self, csv_file_name, agg):
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
		
		# print(type(self.worksheet.at[16796, '5-6 Floor String Pot (8917/18)']))
		# print(self.worksheet.at[16796, '5-6 Floor String Pot (8917/18)'])
		# adding aggregate data column-------------feel free to modify to customize calculation
		if agg == True:
			self.worksheet['Aggregate'] = self.worksheet.iloc[:,1:].astype(float).mean(axis=1, skipna=True)
			print(self.worksheet)

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
		# if needed, further refine the selection of time period
		if 'drying_period' in kwargs: # if make prediction for drying period only, expecting a tuple of datatime obj: (start_dt, end_dt)
			# look up the corresponding idx using the 'drying_period'
			pass
			# check if drying period idx are within the valid time period idx (compare with first and last valid indx)
			#  if not, send the warning msg "Training for drying period only could not be completed"
			pass




		
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


	def train_N_forecast(self, train, forecast_param, use_hyperparam, **kwargs):
		self.forecast_obj = FB_prophet_train_forecast()
		self.forecast_results, self.trained_model = self.forecast_obj.train_forecast(train, forecast_param, use_hyperparam, **kwargs)

		# check if retrain an existing model
		if 'trained_model' in kwargs:
			model_name = 'retrained_model.json'
		else:
			model_name = 'initially_trained_model.json'

		# evaluate the forecast results only when groundtruth data is given
		if 'groundtruth' in kwargs:
			self.eval_results_dict = self.forecast_obj.eval_model(kwargs['groundtruth'], self.forecast_results)
			# log the evaluation results
			for method, result in self.eval_results_dict.items():
				self.logger.info(f"{method}: {result}")

		# save the trained model (see: https://facebook.github.io/prophet/docs/additional_topics.html)
		with open(os.path.sep.join([config.OUTPUT_PATH, model_name]), 'w') as fout:
			json.dump(model_to_json(self.trained_model), fout)


	def plot_results(self, fig_name, trained_model, forecast_results):
		"""
		use the built-in plotting method to plot the forecast results, see: https://facebook.github.io/prophet/docs/quick_start.html#python-api
		"""

		fig = trained_model.plot(forecast_results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
		ax = fig.gca()
		ax.set_xlabel("Time", size=20)
		ax.set_ylabel(self.col_name, size=20)
		fig.savefig(os.path.sep.join([config.OUTPUT_PATH,'{}.png'.format(fig_name)]), dpi=600)

