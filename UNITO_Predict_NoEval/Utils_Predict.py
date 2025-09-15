import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from Data_Preprocessing import normalize
import cv2


def mask_to_gate(pred_list, x_list, subj_list, x_axis, y_axis, gate, gate_pre, path_raw, save_prediction_path, dest, worker = 0, idx = 0, seq = False):
  """
  Prepare data for converting predicted mask to cell-level labels
  args:
    y_list: ground truth mask list
    pred_list: predicted mask list
    x_list: raw image list
    subj_list: subject name list
    x_axis: first gating variable
    y_axis: second gating variable
    gate: current gate
    gate_pre: previous gate
    path_raw: path storing raw data
    save_prediction_path: path for saving the prediction results
    worker: number of workers
    idx: index of current subject in the list
    seq: whether cells from previous gate should be filtered
  """
  raw_img = x_list[worker][idx]
  mask_pred = pred_list[worker][idx]
  subj_path = subj_list[worker][idx]

  # find path for raw tabular data
  substring = os.path.join(f"{dest}/Data/Data_{gate}/Raw_Numpy/")
  subj_path = subj_path.split(substring)[1]
  substring = ".csv.npy"
  subj_path = subj_path.split(substring)[0]

  raw_table = pd.read_csv(path_raw + subj_path + '.csv')
  raw_table = raw_table.reset_index(drop=True)
  
  data_df_pred = get_pred_label(raw_table, x_axis, y_axis, mask_pred, gate, gate_pre, seq)
  data_df_pred.to_csv(os.path.join(f'{save_prediction_path}/{subj_path}.csv'))

  return data_df_pred, subj_path


def get_pred_label(data_df, x_axis, y_axis, mask, gate, gate_pre=None, seq=False):
    """
    Converting mask image to corresponding cell-level prediction label
    argsZ:
      data_df: data containing the cytometric measurement
      x_axis: first gating variable
      y_axis: second gating variable
      mask: predicted mask image
      gate: current gate
      gate_pre: previous gate
      seq: whether cells from previous gate should be filtered
    """

    if seq:
      # prevent reset index fail
      if 'level_0' in data_df.columns:
         data_df = data_df.drop('level_0', axis=1)

      # keep in-gate prediction data for interpolation
      data_df_selected = data_df[data_df[gate_pre]==1].reset_index(drop=True)
      # keep out-gate prediction data for later concat
      data_df_outgate = data_df[data_df[gate_pre]==0].reset_index(drop=True)
      data_df_outgate[x_axis+'_normalized'] = 0
      data_df_outgate[y_axis+'_normalized'] = 0
      data_df_outgate[gate + '_pred'] = 0
    else:
      data_df_selected = data_df
    
    data_df_selected[x_axis+'_normalized'] = data_df_selected[x_axis].copy()
    data_df_selected[x_axis+'_normalized'] = normalize(data_df_selected, x_axis)*100
    data_df_selected[x_axis+'_normalized'] = data_df_selected[x_axis+'_normalized'].round(0).astype(int)

    data_df_selected[y_axis+'_normalized'] = data_df_selected[y_axis].copy()
    data_df_selected[y_axis+'_normalized'] = normalize(data_df_selected, y_axis)*100
    data_df_selected[y_axis+'_normalized'] = data_df_selected[y_axis+'_normalized'].round(0).astype(int)

    # data_df_selected = pd.concat([data_df_selected, data_df[[gate]]], axis = 1)

    index_x = np.linspace(0,100,101).round(0).astype(int).astype(str)
    index_y = np.linspace(0,100,101).round(0).astype(int).astype(str)
    index_x = index_y[::-1] # invert y axis
    df_plot = pd.DataFrame(mask.cpu().numpy().reshape(101,101), index_x, index_y)

    gate_pred = gate + '_pred'
    # data_df_selected[gate_pred] = [int(df_plot.loc[str(a), str(b)]) for (a, b) in zip(data_df_selected[x_axis], data_df_selected[y_axis])]
    pred_label_list = []
    for i in range(data_df_selected.shape[0]):
      a = data_df_selected.loc[i, x_axis+'_normalized']
      b = data_df_selected.loc[i, y_axis+'_normalized']

      if a > 100 or b > 100:
        print('larger than 100') 
        # outlier - label as 0 and continue
        pred_label_list.append(0)
        continue
      pred_label = int(df_plot.loc[str(a), str(b)])
      pred_label_list.append(pred_label)
    data_df_selected[gate_pred] = pred_label_list

    if seq: 
       data_df_recovered = pd.concat([data_df_selected, data_df_outgate])
    else:
       data_df_recovered = data_df_selected
    return data_df_recovered


# def matrix_plot(data_df, x_axis, y_axis, pad_number = 100):
#   """

#   """
#     density = np.zeros((101,101))
#     data_df_selected = data_df[[x_axis, y_axis]]
#     # data_df_selected = data_df_selected.round(2) # round to nearest 0.005
#     data_df_selected_count = data_df_selected.groupby([x_axis, y_axis]).size().reset_index(name="count")

#     coord = data_df_selected_count[[x_axis, y_axis]]
#     # do not normalize event length
#     coord = coord.to_numpy().round(0).astype(int).T
#     coord[0] = 100 - coord[0] # invert position on plot
#     coord = list(zip(coord[0], coord[1]))
#     replace = data_df_selected_count[['count']].to_numpy()
#     for index, value in zip(coord, replace):
#         density[index] = 100 # value + pad_number # this is to make boundary more recognizable for visualization
    
#     index_x = np.linspace(0,1,101).round(2)
#     index_y = np.linspace(0,1,101).round(2)
#     df_plot = pd.DataFrame(density, index_x, index_y)

#     return df_plot

def denoise(img):
    """
    Post processing step to fill holes in the predicted mask
    args:
      img: mask image
    """
    img = img.numpy().astype(dtype=np.uint8)
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if len(contours) <= 1:
       return img
    
    big_contour = max(contours, key=cv2.contourArea)
    little_contour = min(contours, key=cv2.contourArea)

    # get location and area
    area1 = cv2.contourArea(big_contour)
    x1,y1,w1,h1 = cv2.boundingRect(big_contour)
    cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
    area2 = cv2.contourArea(little_contour)
    x2,y2,w2,h2 = cv2.boundingRect(little_contour)
    cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)

    plt.imshow(cv2.fillConvexPoly(img, big_contour, color=255))

    img = np.where(img > 1, 1, 0)
    
    return img


def evaluation(data_df_pred, gate):
    """
    Evaluating gating performance    
    args:
      data_df_pred: predicted dataframe
      gate: the gate name needs to be evaluated
    """
    accuracy = accuracy_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    recall = recall_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    precision = precision_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    f1 = f1_score(data_df_pred[gate], data_df_pred[gate+'_pred'])

    return accuracy, recall, precision, f1
