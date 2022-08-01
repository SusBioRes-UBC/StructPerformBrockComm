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

#print(MAE_dict)
#print(MAE_df)
class Results_Analysis:

    def __init__(self):
        self.Prophet_MAE_df = MAE_df

    def MAE_Line_Plot(self):
        plt.figure(figsize=(10, 8))
        new_MAE_dict = MAE_dict
        new_MAE_dict["Prophet"] =  []
        
        for i in range(self.Prophet_MAE_df.shape[0]):
            new_MAE_dict["Prophet"].append({self.Prophet_MAE_df.iloc[i][0], self.Prophet_MAE_df.iloc[i][1]})

        print(new_MAE_dict)

        for mae in new_MAE_dict:               
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

RA = Results_Analysis()  
RA.MAE_Line_Plot()
    
    

