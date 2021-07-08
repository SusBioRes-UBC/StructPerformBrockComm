"""
This script conducts the time-series analysis for the following target variables for UBC Brock Commons Tallwood House:
	- moisture performance of CLT
	- vertical movement

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


class CLT_perform:
	"""
	This class conducts the time-series analysis for a SINGLE floor
	"""

	def __init__(self, datasheets_path, floor_name):
		# load datasheet
		sheet_path = os.path.sep.join([datasheets_path, floor_name])
		self.worksheet = pd.read_csv(sheet_path)