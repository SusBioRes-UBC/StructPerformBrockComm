"""
This class trains a FB Prophet model, makes the forecast and evaluate the resulting model

Reference: https://www.kaggle.com/kmkarakaya/missing-data-and-time-series-prediction-by-prophet
"""

import brock_comm_config as config
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
		else:
			m = Prophet()
			m.fit(train)

		# make forecast
		future = m.make_future_dataframe(**forecast_params)
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