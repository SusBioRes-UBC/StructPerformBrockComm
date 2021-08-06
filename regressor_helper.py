"""
This helper function prepares the data of regressor(s) to match the format required by FB Prophet
"""

"""
================
Import libraries
================
"""
import pandas as pd


class RegressHelp:

	def prepare_climate_regr(self,raw_clmate_data, **kwargs):
		# load climate data
		raw_climate_data = pd.read_csv(raw_clmate_data, parse_dates=['LOCAL_DATE'])
		retained_climate_data = raw_climate_data[['LOCAL_DATE', 'MEAN_TEMPERATURE','TOTAL_PRECIPITATION']]

		# check if need to expand daily data to hourly 
		if 'convert_day_to_hour_interval' in kwargs:
			# use pd.resample to expand daily data to hourly data
			retained_climate_data.set_index('LOCAL_DATE', inplace=True)
			retained_climate_data = self.retained_climate_data.resample(kwargs['convert_day_to_hour_interval']).pad()

		return retained_climate_data

	def matching_regr_data(self, regr_data, ts_data):
		"""
		matches the timestamp between regressor data and time series data
		Arguments
			- regr_data: a dataframe containing timestep and regressor data
			- ts_data: a dataframe containing timestep and time series data
		Returns:
			- matched_regr_data: part of the regressor data with the common range of timestep 
			- matched_ts_data: part of the time series data with the common range of timestep 
		"""

		# find the interection of timestep between two dataframes

		# slice the two dataframe with the common timestamp range

		# return two adjusted dataframes

		pass