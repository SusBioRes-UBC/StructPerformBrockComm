"""
This class trains a FB Prophet model, makes the forecast and evaluate the resulting model

Reference: https://www.kaggle.com/kmkarakaya/missing-data-and-time-series-prediction-by-prophet
"""

import brock_comm_config as config
from regressor_helper import RegressHelp
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os
import pandas as pd
import json
from prophet.serialize import model_to_json, model_from_json


class FB_prophet_train_forecast:

	def train_forecast(self,train,forecast_params,**kwargs):
		"""
		Arguments:
			- train: training data, df('ds', 'y')
			- forecast_params: a dict of params for .make_future_dataframe(), e.g., {'periods': xx, 'freq': yy}
			- kwargs['regressor_list']: a list containing all the regressor column names
			- kwargs['regressor_trans_func']: a dict of transformation fucntions for regressors, {'regressor_name': func, ...}
		"""
		self.forecast_horizon = forecast_params['periods']

		# check if retrain an existing model, see: https://facebook.github.io/prophet/docs/additional_topics.html#updating-fitted-models
		if 'trained_model' in kwargs:
			# load the model
			with open(os.path.sep.join([config.OUTPUT_PATH,kwargs['trained_model']]), 'r') as fin:
				m = model_from_json(json.load(fin))

			# get params
			res = {}
			for pname in ['k', 'm', 'sigma_obs']:
				res[pname] = m.params[pname][0][0]
			for pname in ['delta', 'beta']:
				res[pname] = m.params[pname][0]
			# retrain model
			m = Prophet().fit(train, init=res)
			# make forecast
			future = m.make_future_dataframe(**forecast_params)
			forecast = m.predict(future)
		else:
			m = Prophet()
			if 'regressor_list' in kwargs:
				#reg_help = RegressHelp()
				for (regressor_name_lst, _) in kwargs['regressor_list']:
					# match the timestep between regressor and time series; regressor_tuple([regressor_col_name1,...,regressor_col_nameN], regressor_dataframe)
					#adjusted_regr, train = reg_help.matching_regr_data(regressor_df, train)
					for regressor_name in regressor_name_lst:
						# add regressor data to training dataframe
						#train[regressor_name] = adjusted_regr[regressor_name].values
						#print(f"there is {sum(train[regressor_name].isna())} NaN in {train[regressor_name]}")
						#print(f"tail values for {regressor_name} is {train[regressor_name].tail()}")

						# add regressor to the model
						m.add_regressor(regressor_name)

				#print(f"train data looks like: {train.tail}")
				#print(f"index name of train is {train.index}")

				# train the model
				m.fit(train) # 'train' should contain already-transformed regressor values
				# make forecast
				future = m.make_future_dataframe(**forecast_params)
				#print(f"what does future look like? {future.tail()}")
				
				for (regressor_name_lst, _) in kwargs['regressor_list']:	
					for regressor_name in regressor_name_lst:			
						if 'regressor_trans_func' in kwargs: # dict{'regressor_name1': func1, ..., 'regressor_nameN': funcN}
							# apply the same 'ds'-based transformation function (used to transform training data) to the future regressor values 
							# [caution] the code below (apply func) has NOT been tested yet as of Aug.17, 2021
							future[regressor_name] = future['ds'].apply(kwargs['regressor_trans_func'][regressor_name])
						else:
							# use the historical data (last "forecast_params['periods']" data points) --> only works properly for in-sample prediction, need to provide a kwargs['test'] <<-this is not going to work, as index of test_df is not datetime obj
							#future[regressor_name] = future['ds'].apply(lambda x: kwargs['test'].loc[x,regressor_name])
							#print(f"future values for {regressor_name} is {future[regressor_name].tail()}")
							n_future_regr_data = forecast_params['periods'] 
							# concat the train df with designated amt of regressor data from a separate df. [Caution]: this will not work if: (1) no separate dataset for future values of regr is avaiable or (2) # of data points in that dataset is less than those needed for forecast
							future[regressor_name] = pd.concat([train[regressor_name],kwargs["regr_future"][regressor_name][:n_future_regr_data]], axis=0).values

				print(f"what does future look like? {future.sample(24)}")

			else: 
				# train the model directly, if no regressor is provider
				m.fit(train)
				# make forecast
				future = m.make_future_dataframe(**forecast_params)
			
			# make prediction
			forecast = m.predict(future)

		#forecast = forecast[['ds', 'yhat']]
		#forecast = forecast[-forecast_horizon:] 
		#print(f"shape of forecast obj: {forecast.shape}")
		print(f"tail of forecast results: {forecast[['ds', 'yhat']].tail()}")
		return forecast, m


	def eval_model(self, groundtruth, forecast_results):
		"""
		Argument:
			- groundtruth: a df with same shape train_df, typically df('ds', 'y')
			- forecast_results: a df with a shape of (n,22), important info ('ds, 'yhat', 'yhat_lower', 'yhat_higher')
		"""
		print(f"groundtruth shape is {groundtruth['y'].shape}")
		print(f"forecast_results is {forecast_results['yhat'].shape}")

		return {'MAE': mean_absolute_error(groundtruth['y'], forecast_results['yhat'][-self.forecast_horizon:])}