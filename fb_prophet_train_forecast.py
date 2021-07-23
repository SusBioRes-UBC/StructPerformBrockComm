"""
This class trains a FB Prophet model, makes the forecast and evaluate the resulting model

Reference: https://www.kaggle.com/kmkarakaya/missing-data-and-time-series-prediction-by-prophet
"""

from prophet import Prophet
from sklearn.metrics import mean_absolute_error

class FB_prophet_train_forecast:

	def train_forecast(self,train,forecast_params):
		"""
		Arguments:
			- train: training data, df('ds', 'y')
			- forecast_params: a dict of params for .make_future_dataframe(), e.g., {'periods': xx, 'freq': yy}
		"""

		self.forecast_horizon = forecast_params['periods']
		m = Prophet()
		m.fit(train)
		future = m.make_future_dataframe(**forecast_params)
		forecast = m.predict(future)
		#forecast = forecast[['ds', 'yhat']]
		#forecast = forecast[-forecast_horizon:] 
		#print(f"shape of forecast obj: {forecast.shape}")
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