"""
This helper function prepares the data of regressor(s) to match the format required by Darts
"""
"""
================
Import libraries
================
"""
import pandas as pd
from sklearn.impute import SimpleImputer

class RegressHelp:
	"""
	May need to include an 'impute' method later
	"""

	def prepare_climate_regr(self,raw_clmate_data, **kwargs):
		# load climate data
		raw_climate_data = pd.read_csv(raw_clmate_data, parse_dates=['LOCAL_DATE'])
		retained_climate_data = raw_climate_data[['LOCAL_DATE', 'MEAN_TEMPERATURE','TOTAL_PRECIPITATION']]
		retained_climate_data.set_index('LOCAL_DATE', inplace=True)

		# imputation to remove NaN
		if 'impute' in kwargs:
			for col in ['MEAN_TEMPERATURE','TOTAL_PRECIPITATION']:
				my_imputer = SimpleImputer(strategy=kwargs['impute'])
				y = retained_climate_data[col].values
				y = y.reshape(-1, 1)
				y_imputed = my_imputer.fit_transform(y).tolist()
				y_imputed = [y[0] for y in y_imputed]

				retained_climate_data['y_imputed'] = y_imputed
				retained_climate_data.drop(columns=[col], inplace=True)
				retained_climate_data.rename(columns={'y_imputed': col}, inplace=True)

		# check if need to expand daily data to hourly 
		if 'convert_day_to_hour_interval' in kwargs:
			# use pd.resample to expand daily data to hourly data
			retained_climate_data = retained_climate_data.resample(kwargs['convert_day_to_hour_interval']).pad()

		# create a "ds" column
		retained_climate_data['ds'] = retained_climate_data.index
		retained_climate_data.reset_index()

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
		return regr_data.copy().loc[common_idx[0]: common_idx[-1]], ts_data.copy().loc[common_idx[0]: common_idx[-1]]
