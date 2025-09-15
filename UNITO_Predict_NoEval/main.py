from Validation_Recon_Plot_Single import plot_all
from Data_Preprocessing import process_table, train_test_val_split
from Predict import UNITO_gating, evaluation
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# For reproducibility
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

# setting gates
gating = pd.read_csv('./gating_structure.csv')
gate_pre_list = [x + '_pred' if str(x) != 'nan' else x for x in list(gating.Parent_Gate)]
gate_pre_list[0] = None # the first gate does not have parent gate
gate_list = list(gating.Gate)
x_axis_list = list(gating.X_axis)
y_axis_list = list(gating.Y_axis)
path2_lastgate_pred_list = ['./prediction/' for x in range(len(gate_list))]
path2_lastgate_pred_list[0] = './Raw_Data_pred/' # the first gate should take data from raw folder

# hyperparameter
# device = 'cuda' if torch.cuda.is_available() else 'cpu' 
device = 'mps'
n_worker = 0
convex = True

# define path
dest = '.' # change depending on your needs
save_data_img_path = f'{dest}/Data'
save_figure_path = f'{dest}/figures'
save_prediction_path = f'{dest}/prediction'

# make directory
if not os.path.exists(save_data_img_path):
    os.mkdir(save_data_img_path)
if not os.path.exists(save_figure_path):
    os.mkdir(save_figure_path)
if not os.path.exists(save_prediction_path):
    os.mkdir(save_prediction_path)
    
###########################################################
# UNITO
###########################################################
hyperparameter_df = pd.DataFrame(columns = ['gate','learning_rate','batch_size'])

for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):

    print(f"start UNITO for {gate}")

    # 1. preprocess training data
    # pred_path = './Raw_Data_pred'
    process_table(x_axis, y_axis, gate_pre, gate, path_raw, convex, seq = (gate_pre!=None), dest = dest)
    train_test_val_split(gate, path_raw, dest, './pred')

    # 2. predict
    model_path = f'{dest}/model/{gate}_model.pt'
    data_df_pred = UNITO_gating(model_path, x_axis, y_axis, gate, path_raw, n_worker, device, save_prediction_path, dest, seq = (gate_pre!=None), gate_pre=gate_pre)

    # # 3. Evaluation
    # accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
    # print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precition:{precision}, f1 score:{f1}")

    # 4. Plot gating results
    plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
    print("All UNITO prediction visualization saved")

print("Seqential autogating prediction finished")