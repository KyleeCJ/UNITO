import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def evaluation(data_df_pred, gate):
    """
    Calculate accuracy, recall, precision, and f1 score for a single subject, single gate
    args:
        data_df_pred: dataframe contains the predicted cell-level label and ground truth label
        gate: gate needs to be evaluated
    """
    accuracy = accuracy_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    recall = recall_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    precision = precision_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    f1 = f1_score(data_df_pred[gate], data_df_pred[gate+'_pred'])

    return accuracy, recall, precision, f1

def evaluate_gate(root, gate):
    """
    calculate evaluation score for one gate over all subjects
    args:
        root: root path containing all subjects data
        gate: gate needs to be evaluated
    """
    # get all subjects from root directory
    subjects = os.listdir(root)
    subjects = [x for x in subjects if 'csv' in x]

    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    # iterate over all subjects
    for subject in subjects:
        df = pd.read_csv(os.path.join(root, subject))
        accuracy, recall, precision, f1 = evaluation(df, gate)

        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)

    # calculate average evaluation scores
    accuracy_avg = sum(accuracy_list) / len(accuracy_list)
    recall_avg = sum(recall_list) / len(recall_list)
    precision_avg = sum(precision_list) / len(precision_list)
    f1_avg = sum(f1_list) / len(f1_list)

    return accuracy_avg, recall_avg, precision_avg, f1_avg
