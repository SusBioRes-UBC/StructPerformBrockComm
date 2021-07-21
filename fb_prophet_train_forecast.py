"""
This class trains a FB Prophet model, makes the forecast and evaluate the resulting model

Reference: https://www.kaggle.com/kmkarakaya/missing-data-and-time-series-prediction-by-prophet
"""

from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error

class FB_prophet_train_forecast:

	def train_forecast(self,train,forecast_params,forecast_horizon):
		"""
		Arguments:
			- train: training data, df
			- forecast_params: a dict of params for .make_future_dataframe(), e.g., {'periods': xx, 'freq': yy}
			- forecast_horizon: 
		"""

		m = Prophet()
		m.fit(train)
		# Python
		future = m.make_future_dataframe(**forecast_params)
		forecast = m.predict(future)
		#forecast = forecast[['ds', 'yhat']]
		#forecast = forecast[-forecast_horizon:] 
		return forecast

	def eval_model(self, groundtruth, forecast_results):
		return {'MAE': mean_absolute_error(groundtruth, forecast_results['yhat'])}