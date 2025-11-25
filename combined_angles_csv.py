import pandas as pd
import numpy as np
import os
from pathlib import Path
from sys import path


pd.options.mode.use_inf_as_na = True

ANGLES_DIR = Path(path[0] + "\\angles\\parameters")
columns = [
            'angle','true_dx','true_dy',
            "0_0_dx","0_0_dy","0_1_dx","0_1_dy","0_2_dx","0_2_dy",
            "1_0_dx","1_0_dy","1_1_dx","1_1_dy","1_2_dx","1_2_dy",
            "2_0_dx","2_0_dy","2_1_dx","2_1_dy","2_2_dx","2_2_dy"
        ]


dataframes = []


for file in os.listdir(ANGLES_DIR):
    if file.endswith('.csv'):  
        file_path = ANGLES_DIR / file
        df = pd.read_csv(file_path)
            
        df_features = df.loc[:, columns]

        # Формирование таргетных столбцов
        df_features.pop("true_dx")
        df_features.pop("true_dy")
        angle = df_features.pop("angle")
        df_features.insert(len(df_features.columns), "angle", angle)
        
        dataframes.append(df_features)

if dataframes:
    all_data = pd.concat(dataframes, ignore_index=True)
else:
    all_data = pd.DataFrame(columns=columns)
    
all_data.fillna(0, inplace=True)

print(all_data)

csv_path = Path(path[0] + "\\angles\\combined_data_angle.csv")
all_data.to_csv(csv_path, index=False, encoding='utf8')