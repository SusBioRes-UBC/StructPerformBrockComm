"""
This script conducts the results analysis for the following target variables for UBC Brock Commons Tallwood House:
    - moisture performance of CLT
    - vertical movement
Author: Leo Sun, Manqin Cai
"""
"""
================
Import libraries
================
"""
import matplotlib.pyplot as plt
import plot_config as config
import os

class Results_Analysis:

    def __init__(self, **kwargs):
        self.Prophet_MAE_df = kwargs['MAE_df']
        self.Darts_MAE_dict = kwargs['MAE_dict']
        self.Prophet_forecast_results_dict = kwargs['forecast_dict']
        self.Darts_forecast_results_dict = kwargs['forecasts_all_dict']
        self.groundtruth_dict = kwargs['groundtruth_dict']

    def MAE_Line_Plot(self):
        plt.figure(figsize=(10, 8))
        MAE_all_dict = self.Darts_MAE_dict
        MAE_all_dict["Prophet"] =  []
        
        for i in range(self.Prophet_MAE_df.shape[0]):
            d = {self.Prophet_MAE_df.iloc[i][0]: self.Prophet_MAE_df.iloc[i][1]}
            MAE_all_dict["Prophet"].append(d)

        print(MAE_all_dict)

        for mae in MAE_all_dict:               
            MAE_floor_list = []
            MAE_value_list = []
            for m_d in self.Darts_MAE_dict[mae]:
                floor = list(m_d.keys())[0]
                val = list(m_d.values())[0]
                MAE_floor_list.append(floor)
                MAE_value_list.append(val)
            
            plt.plot(MAE_floor_list, MAE_value_list,label=mae)
            
        plt.xlabel("floor")
        plt.ylabel("MAE")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.sep.join([config.OUTPUT_PATH,'{}.png'.format("MAE")]), dpi=600)
        plt.show()

    def Forecasts_Line_Plot(self):
        #print(self.Prophet_forecast_results_dict)
        #print(self.Darts_forecast_results_dict)
        plt.figure(figsize=(10, 8))
        all_forecasts = {}

        for floorkey in self.Darts_forecast_results_dict.keys():
            all_forecasts[floorkey] = self.Darts_forecast_results_dict[floorkey]
            all_forecasts[floorkey]["Prophet"] = self.Prophet_forecast_results_dict[floorkey]

        print(all_forecasts)

        for fkey, m_df_dict in all_forecasts.items():
            plt.plot(self.groundtruth_dict[fkey]['ds'], self.groundtruth_dict[fkey]['y'], label = 'groundtruth')
            for modelkey, results_df in m_df_dict.items():
                plt.plot(results_df['ds'], results_df['y'], label = modelkey)

            #groundtruth to add
            plt.xlabel("timestamp")
            plt.ylabel("yhat")
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.sep.join([config.OUTPUT_PATH, str(fkey) + " forecasts.png"]), dpi=600)
            plt.show()

  
    
