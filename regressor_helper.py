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
	"""
	May need to include an 'impute' method later
	"""

	def prepare_climate_regr(self,raw_clmate_data, **kwargs):
		# load climate data
		raw_climate_data = pd.read_csv(raw_clmate_data, parse_dates=['LOCAL_DATE'])
		retained_climate_data = raw_climate_data[['LOCAL_DATE', 'MEAN_TEMPERATURE','TOTAL_PRECIPITATION']]
		retained_climate_data.set_index('LOCAL_DATE', inplace=True)

		# check if need to expand daily data to hourly 
		if 'convert_day_to_hour_interval' in kwargs:
			# use pd.resample to expand daily data to hourly data
			retained_climate_data = retained_climate_data.resample(kwargs['convert_day_to_hour_interval']).pad()

		# create a "ds" column
		retained_climate_data['ds'] = retained_climate_data.index

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
		# set index of ts_data to dateime
		ts_data["INDEX"] = ts_data['ds'].apply(lambda x: x)
		ts_data.set_index("INDEX", inplace=True)

		# find the interection of timestep between two dataframes
		common_idx = ts_data.index.intersection(regr_data.index)
		print(f"common idex starts with: {common_idx[0]} and ends with {common_idx[-1]}")

		# return two adjusted dataframes with the common timestamp range
		return ts_data.copy()[common_idx[0]: common_idx[-1]], regr_data.copy()[common_idx[0]: common_idx[-1]]