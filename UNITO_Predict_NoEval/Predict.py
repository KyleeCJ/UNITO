import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNITO_Model import UNITO
from Dataset import dataset
from Utils_Predict import *
from Data_Preprocessing import *

def UNITO_gating(model_path, x_axis, y_axis, gate, path_raw, num_workers, device, save_prediction_path, dest, seq = False, gate_pre = None):
  """
  Performs UNITO auto-gating
  args:
    model_path: the path for storing models
    x_axis: first gating variable
    y_axis: second gating variable
    gate: current gate
    path_raw: path storing raw data
    num_workers: number of worker for pytorch setting
    device: whether use GPU or not
    save_prediction_path: path for save predition results
    seq: whether cells from previous gate should be filtered
    gate_pre: previous gate  
  """

  # in sequential predicting, the path_raw is the path for prediction of last gate
  PATH = os.path.join(model_path)
  model = UNITO().to(device)
  model.load_state_dict(torch.load(PATH, map_location=device))
  model.eval()

  test_transforms = A.Compose(
      [
        ToTensorV2(),
      ],
  )

  path_val = pd.read_csv(f"{dest}/Data/Data_{gate}/pred/subj.csv")

  val_ds = dataset(path_val, test_transforms)
  val_loader = DataLoader(val_ds, batch_size = path_val.shape[0], num_workers = num_workers, pin_memory = True)

  preds_list, x_list, subj_list = predict_visualization(val_loader, model, device)

  for ind in range(path_val.shape[0]):

      data_df_pred, subj_path = mask_to_gate(preds_list, x_list, subj_list, x_axis, y_axis, gate, gate_pre, path_raw, save_prediction_path, dest, worker = 0, idx = ind, seq = seq)

  print("UNITO prediction finished")

  return data_df_pred

def predict_visualization(loader, model, device="mps"):
  """
  Run UNITO model for prediction
  args:
    loader: loader for loading data
    model: UNITO model for prediction
    device: whether use GPU
  """
  model.eval()
  preds_list = []
  x_list = []
  subj_list = []
  for idx, (x,subj) in enumerate(loader):
    x = x.type(torch.float32)
    x = x.to(device=device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
    preds_list.append(preds)
    x_list.append(x.unsqueeze(1))
    subj_list.append(subj)

  return preds_list, x_list, subj_list


def clean_val_path(subj_path, gate):
    """
    formulate the path for reading purpose
    args:
      subj_path: path for subject data
      gate: gate name
    """
    # find path for raw tabular data
    if 'Raw' in subj_path:
      substring = os.path.join(f"./Data_image/Data_{gate}/Raw_Numpy/")
    else:
      substring = os.path.join(f"./Data_image/Data_{gate}/Mask_Numpy/")
    subj_path = subj_path.split(substring)[1]
    substring = ".csv.npy"
    subj_path = subj_path.split(substring)[0]
    return subj_path
