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
import pandas as pd
import os
from Prophet.prediction import MAE_df 
from Darts.Final_Pipeline import MAE_dict 

class Results_Analysis:
    Prophet_MAE_df = MAE_df

    def MAE_Line_Plot(self, **kwargs):
        plt.figure(figsize=(10, 8))
        for mae in MAE_dict:               
            MAE_floor_list = []
            MAE_value_list = []
            for m_d in MAE_dict[mae]:
                floor = list(m_d.keys())[0]
                val = list(m_d.values())[0]
                MAE_floor_list.append(floor)
                MAE_value_list.append(val)
            
            plt.plot(MAE_floor_list, MAE_value_list,label=mae)
            
        plt.xlabel("floor")
        plt.ylabel("MAE")
        plt.legend(loc='best')
        plt.savefig(os.path.sep.join([config.OUTPUT_PATH,'{}.png'.format("MAE")]), dpi=600)
        plt.show()
    
    def Forecasts_Line_Plot(self, **kwargs):
        print("")

