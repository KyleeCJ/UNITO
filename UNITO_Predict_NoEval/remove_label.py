import pandas as pd
import os

df_list = os.listdir('./Raw_Data_pred')
for df in df_list:
    if 'csv' in df:
        df_path = os.path.join('./Raw_Data_pred', df)
        df_ = pd.read_csv(df_path)
        columns_to_remove = ['Unnamed: 0', 'Single_cell_gate_1', 'Single_cell_gate_2',
       'Granulocyte', 'Neutrophil', 'Eosinophil', 'Mononuclear', 'T_cell',
       'B_cell', 'No_T_Non_B_cell', 'Categorical']
        df_ = df_.drop(columns=columns_to_remove, errors='ignore')
        df_.to_csv(df_path, index=False)
