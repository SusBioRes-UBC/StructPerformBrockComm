"""
This script ties all the data cleaning, training, forecasting, results collection and results analysis from all models
Author: Leo Sun, Manqin Cai
"""
# import the pipelines 
from Darts.Final_Pipeline import Darts_Pipeline
from Prophet.prediction import Prophet_Pipeline
from Results_Analysis import Results_Analysis

# run the pipelines and get the MAE and forecast results
MAE_dict, forecasts_all_dict = Darts_Pipeline()
MAE_df, forecast_dict, groundtruth_dict = Prophet_Pipeline()
#print(MAE_dict)
#print(forecasts_all_dict)
#print(MAE_df)
#print(forecast_dict)
#print(groundtruth_dict)

# do results analysis and plotting
RA = Results_Analysis(MAE_dict = MAE_dict, forecasts_all_dict = forecasts_all_dict, 
                      MAE_df = MAE_df, forecast_dict = forecast_dict, groundtruth_dict = groundtruth_dict)  
RA.MAE_Line_Plot()
RA.Forecasts_Line_Plot()