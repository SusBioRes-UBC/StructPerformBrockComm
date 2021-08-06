"""
This class trains a FB Prophet model, makes the forecast and evaluate the resulting model

Reference: https://www.kaggle.com/kmkarakaya/missing-data-and-time-series-prediction-by-prophet
"""

import brock_comm_config as config
from regressor_helper import RegressHelp
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os
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
				for regressor_tuple in kwargs['regressor_list']:
					# match the timestep between regressor and time series; regressor_tuple(regressor_name, regressor_dataframe)
					pass
					# add regressor data to training dataframe
					#train[regressor_tuple[0]] = matched_regr_data
					# add regressor to the model
					m.add_regressor(regressor_tuple[0])

				# train the model
				m.fit(train) # 'train' should contain already-transformed regressor values
				# make forecast
				future = m.make_future_dataframe(**forecast_params)
				for regressor in kwargs['regressor_list']:
					# apply the same transformation function (used to transform training data) to the future regressor values 
					future[regressor] = future['ds'].apply(kwargs['regressor_trans_func'][regressor])
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