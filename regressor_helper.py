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
		self.raw_climate_data = pd.read_csv(raw_clmate_data, parse_dates=['LOCAL_DATE'])
		self.retained_climate_data = self.raw_climate_data[['LOCAL_DATE', 'MEAN_TEMPERATURE','TOTAL_PRECIPITATION']]

		# check if need to expand daily data to hourly 
		if 'convert_day_to_hour_interval' in kwargs:
			# use pd.resample to expand daily data to hourly data
			self.retained_climate_data.set_index('LOCAL_DATE', inplace=True)
			self.retained_climate_data = self.retained_climate_data.resample(kwargs['convert_day_to_hour_interval']).pad()
