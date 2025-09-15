import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datashader as ds
from datashader.mpl_ext import dsshow
from scipy.spatial import ConvexHull

def plot_all(gate1, gate2, x_axis, y_axis, raw_path, figure_path):
    """
    Plot gating results for all samples
    args:
        gate1: previous gates that needs to filter
        gate2: current gate
        x_axis: first gating variable
        y_axis: second gating variable
        raw_path: path containing raw data
        figure_path: path to save figures
    """

    if not os.path.exists(f"{figure_path}/Figure_{gate2}"):
        os.mkdir(f"{figure_path}/Figure_{gate2}")

    path_val = pd.read_csv(f"./Data/Data_{gate2}/pred/subj.csv")
    # find path for raw tabular data
    for subject in path_val.Image:
        plot_one(gate1, gate2, x_axis, y_axis, subject, raw_path, figure_path)
        
def plot_one(gate1, gate2, x_axis, y_axis, subject, raw_path, figure_path):
    """
    Plot a single sample with target gate
    args:
        gate1: previous gates that needs to filter
        gate2: current gate
        x_axis: first gating variable
        y_axis: second gating variable
        subject: sample name
        raw_path: path containing raw data
        figure_path: path to save figures
    """
    substring = f"./Data/Data_{gate2}/Raw_Numpy/"   
    subject = subject.split(substring)[1]
    substring = ".npy"
    subject = subject.split(substring)[0]
    
    data_table = pd.read_csv(f'{raw_path}/{subject}')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # ###########################################################
    # # True
    # ###########################################################

    # data = data_table.copy()
    # if gate1 != None:
    #     data = data[data[gate1]==1]
    
    # in_gate = data[data[gate2] == 1]
    # out_gate = data[data[gate2] == 0]
    # in_gate = in_gate[[x_axis, y_axis]].to_numpy()
    # out_gate = out_gate[[x_axis, y_axis]].to_numpy()

    # if in_gate.shape[0] > 3:
    #     hull = ConvexHull(in_gate)
    #     for simplex in hull.simplices:
    #         ax[0].plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

    #     density_plot = dsshow(
    #         data,
    #         ds.Point(x_axis, y_axis),
    #         ds.count(),
    #         vmin=0,
    #         vmax=300,
    #         norm='linear',
    #         cmap='jet',
    #         aspect='auto',
    #         ax=ax[0],
    #     )

    # # cbar = plt.colorbar(density_plot)
    # # cbar.set_label('Number of cells in pixel')
    # ax[0].set_xlabel(x_axis)
    # ax[0].set_ylabel(y_axis)

    # ax[0].set_title("Manual Gating")


    ###########################################################
    # predict
    ###########################################################
    
    x_axis_pred = x_axis
    y_axis_pred = y_axis

    if gate1 != None:
        gate1_pred = gate1
    gate2_pred = gate2 + '_pred'

    data_table = pd.read_csv(f'./prediction/{subject}')

    data = data_table.copy()
    if gate1 != None:
        data = data[data[gate1_pred]==1]
    in_gate = data[data[gate2_pred] == 1]
    out_gate = data[data[gate2_pred] == 0]
    in_gate = in_gate[[x_axis_pred, y_axis_pred]].to_numpy()
    out_gate = out_gate[[x_axis_pred, y_axis_pred]].to_numpy()

    if in_gate.shape[0] > 3:
        hull = ConvexHull(in_gate)
        for simplex in hull.simplices:
            ax.plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

        density_plot = dsshow(
            data,
            ds.Point(x_axis_pred, y_axis_pred),
            ds.count(),
            vmin=0,
            vmax=300,
            norm='linear',
            cmap='jet',
            aspect='auto',
            ax=ax,
        )

    # cbar = plt.colorbar(density_plot)
    # cbar.set_label('Number of cells in pixel')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    ax.set_title("UNITO Auto-gating")

    # save figure
    plt.savefig(f'{figure_path}/Figure_{gate2}/Recon_Sequential_{subject}.png')
    plt.close()

