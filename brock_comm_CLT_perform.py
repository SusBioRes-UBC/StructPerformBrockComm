"""
This script conducts the time-series analysis for the following target variables for UBC Brock Commons Tallwood House:
	- moisture performance of CLT
	- vertical movement (??)

Author: Qingshi

References:
- UBC Brock Commons Structural Performance Report Sept 2020: https://sustain.ubc.ca/sites/default/files/UBC%20Brock%20Commons%20Structural%20Performance%20Report%20Sept%202020.pdf 
"""

"""
================
Import libraries
================
"""
import pmdarima as pm
import pandas as pd
import brock_comm_config as config
import os
from datetime import datetime as dt
import logging


class CLT_perform:
	"""
	This class conducts the time-series analysis for a SINGLE floor
	"""

	def __init__(self, floor_file_name):
		# load datasheet
		sheet_path = os.path.sep.join([config.DATASHEETS_PATH, floor_file_name])
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
		file_handler = logging.FileHandler('log_LCA_calculation.log')
		formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
		file_handler.setFormatter(formatter)

		# add file handler to logger
		self.logger.addHandler(file_handler)  


	def preprocess(self):
		# data cleaning
		# - remove white space for column names
		self.worksheet.columns = [x.strip() for x in list(self.worksheet.columns)]
		# - create two new columns to store Date and Time separately
		self.worksheet['Date'] = self.worksheet["DateTime"].apply(lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S%z").date())
		self.worksheet['Time'] = self.worksheet["DateTime"].apply(lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S%z").time())

		# 



