import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sn
import scipy
import cv2

def normalize(data, column):
    """
    normalize the cytometric measurement
    args:
        data: dataframe containing all cytometric measurement
        column: column name that needs to be normalized
    """
    df_normalize = data[column]
    min = df_normalize.min()
    max = df_normalize.max()
    df_normalize = (df_normalize-min)/(max-min)
    return df_normalize

def matrix_plot(data_df_selected, x_axis, y_axis, pad_number = 0):
    """
    draw image based on selected cytometric measurement
    args:
        data: dataframe containing all cytometric measurement
        x_axis: first gating variable
        y_axis: second gating variable
        pad_number: allows uer to augment the cell-pixel to distinguish from background, set to 0 by default
    """
    density = np.zeros((101,101))

    data_df_selected = data_df_selected.round(0)
    data_df_selected_count = data_df_selected.groupby([x_axis, y_axis]).size().reset_index(name="count")

    coord = data_df_selected_count[[x_axis, y_axis]]
    coord = coord.to_numpy().round(0).astype(int).T
    coord[0] = 100 - coord[0] # invert position on plot
    coord = list(zip(coord[0], coord[1]))
    replace = data_df_selected_count[['count']].to_numpy()
    for index, value in zip(coord, replace):
        density[index] = value + pad_number
    
    index_x = np.linspace(0,100,101).round(2)
    index_y = np.linspace(0,100,101).round(2)
    df_plot = pd.DataFrame(density, index_x, index_y)

    return df_plot

def export_matrix(file_name, x_axis, y_axis, gate_pre, gate, path_raw, convex, seq = False, dest = '.'):
    """
    Converting column value from cytometric data to images
    args
        file_name: the file needs to be processed
        x_axis: first gating variable
        y_axis: second gating variable
        gate_pre: previous gate
        gate: current gate
        convex: control mask processing
        seq: whether cells from previous gate should be filtered
        raw_path: path storing raw data
    """
    data_df = pd.read_csv(os.path.join(path_raw, file_name))
    if seq:
        data_df = data_df[data_df[gate_pre]==1]

    # data_df_selected = data_df[[x_axis, y_axis, gate]]
    data_df_selected = data_df[[x_axis, y_axis]]
    data_df_selected[x_axis] = normalize(data_df_selected, x_axis)
    data_df_selected[x_axis] = data_df_selected[x_axis]*100
    data_df_selected[y_axis] = normalize(data_df_selected, y_axis)
    data_df_selected[y_axis] = data_df_selected[y_axis]*100

    fig = plt.figure()
    df_plot = matrix_plot(data_df_selected, x_axis, y_axis, 0)
    sn.heatmap(df_plot, vmax = df_plot.max().max()/2, vmin = df_plot.min().min()/2)
    plt.savefig(os.path.join(f'{dest}/Data/Data_{gate}/Raw_PNG/', file_name+'.png'))
    np.save(os.path.join(f'{dest}/Data/Data_{gate}/Raw_Numpy/', file_name+'.npy'), df_plot)
    plt.close()
    
    # fig = plt.figure()
    # data_df_masked_2 = data_df_selected[data_df_selected[gate]==1]
    # df_plot = matrix_plot(data_df_masked_2, x_axis, y_axis, 0)
    # df_plot = df_plot.applymap(lambda x: 1 if x != 0 else 0)
    # # check if there is points in gate
    # df_plot = df_plot.to_numpy()
    # if np.sum(df_plot) > 3:
    #     df_plot = fill_hull(df_plot, convex)
    # sn.heatmap(df_plot, vmax = df_plot.max().max()/2, vmin = df_plot.min().min()/2)
    # plt.savefig(os.path.join(f'{dest}/Data/Data_{gate}/Mask_PNG/', file_name+'.png'))
    # np.save(os.path.join(f'{dest}/Data/Data_{gate}/Mask_Numpy/', file_name+'.npy'), df_plot)
    # plt.close()

def process_table(x_axis, y_axis, gate_pre, gate, directory, convex, seq = False, dest = '.'):   
    """
    Preparing images for deep learning model
    args:
        x_axis: first gating variable
        y_axis: second gating variable
        gate_pre: previous gate
        gate: current gate
        directory: path storing raw data
        convex: control mask processing
        seq: whether cells from previous gate should be filtered
    """
    if not os.path.exists(f'{dest}/Data/Data_{gate}'):
        os.mkdir(f"{dest}/Data/Data_{gate}")
        os.mkdir(f"{dest}/Data/Data_{gate}/Mask_Numpy")
        os.mkdir(f"{dest}/Data/Data_{gate}/Mask_PNG")
        os.mkdir(f"{dest}/Data/Data_{gate}/Raw_Numpy")
        os.mkdir(f"{dest}/Data/Data_{gate}/Raw_PNG")

    # iterate over files in that directory
    name_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if 'csv' in filename: 
                name_list.append(filename)
    name_list_df = pd.DataFrame(name_list, columns =['subject']) 
    name_list_df['subject_id'] = [x.split('.')[0] for x in name_list_df['subject']] # remove .csv
    
    # process the baseline subject
    for filename in name_list:
        print(f"processing {filename}")
        export_matrix(filename, x_axis, y_axis, gate_pre, gate, directory, convex, seq, dest)
    print("process table finished")


def filter(path_list):
    """
    keep only npy files in path list, just in case there are hidden sync files
    args:
        path_list: path containing the data
    """
    path_ = []
    for path in path_list:
        if 'npy' in path:
            path_.append(path)
    return path_

def train_test_val_split(gate, path, dest = '.', train_pred = 'train'):
    """
    Prepare the subject list for training and prediction
    args:
        gate: the gate name that needs to be gated
    """
    if not os.path.exists(f"{dest}/Data/Data_{gate}/{train_pred}"):
        os.mkdir(f"{dest}/Data/Data_{gate}/{train_pred}")

    subj_list = os.listdir(path)
    subj_list = [x+'.npy' for x in subj_list if 'csv' in x]

    imgs_ = [f"{dest}/Data/Data_{gate}/Raw_Numpy/"+x for x in subj_list]
    masks_ = [f"{dest}/Data/Data_{gate}/Mask_Numpy/"+x for x in subj_list]

    imgs = filter(imgs_)
    masks = filter(masks_)
    path = pd.DataFrame(list(zip(imgs, masks)), columns = ['Image','Mask'])

    path.to_csv(f"{dest}/Data/Data_{gate}/{train_pred}/subj.csv", index=False)


def fill_hull(image, convex=True):
    """
    Compute the filled region of the given binary image.
    If convex is True, compute the convex hull and return a mask of the filled hull.
    If convex is False, only fill in the blank pixels within the existing boundaries.
    
    Adapted from:
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    """
    if convex:
        # Convex hull filling
        points = np.argwhere(image).astype(np.int16)
        hull = scipy.spatial.ConvexHull(points)
        convex_points = []
        for vertex in hull.vertices:
            convex_points.append(points[vertex].astype('int64'))
        convex_points = np.array(convex_points)
        convex_points[:, [1, 0]] = convex_points[:, [0, 1]]

        a, b = image.shape
        black_frame = np.zeros([a, b], dtype=np.uint8)
        cv2.fillPoly(black_frame, pts=[convex_points], color=(255, 255, 255))
        black_frame[black_frame == 255] = 1
    else:
        # Filling within the existing object boundaries
        black_frame = image.copy().astype(np.uint8)
        contours, _ = cv2.findContours(black_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(black_frame, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        black_frame[black_frame == 255] = 1

    return black_frame