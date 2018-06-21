""" coding: utf-8
Created by rsanchez on 21/06/2018
Este proyecto ha sido desarrollado en la Gerencia de Operaciones de CENACE
Mateo633
"""

import numpy as np
import pandas as pd
# from hmmlearn.hmm import GaussianHMM
# from sklearn.externals import joblib
from collections import Counter
from my_lib.hmm import hmm_util as hmm_u
from my_lib.PI_connection import pi_connect as pi


def day_cluster_matrix(hmm_model, df_y):
    n_comp = hmm_model.n_components
    dict_cl_week = dict()
    sp = min(0.005 * len(df_y.index), 4)

    for n in range(n_comp):
        mask = df_y["hidden_states"].isin([n])
        df_aux = df_y[mask]
        n_members = len(df_aux.index)
        df_aux.index = pd.to_datetime(df_aux.index)
        df_aux.index = [x.weekday_name for x in df_aux.index]

        if n_members > sp:
            c_dict = Counter(list(df_aux.index))
            for c in c_dict:
                if c_dict[c] > n_members * 0.15:
                    dict_cl_week = append_in_list(dict_cl_week, c, n)
                else:
                    dict_cl_week = append_in_list(dict_cl_week, "sp_" + c, n)
        else:
            dict_cl_week = append_in_list(dict_cl_week, "special", n)

    return dict_cl_week


def append_in_list(dict_obj, key, value):
    if key in dict_obj.keys():
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [value]
    dict_obj[key] = list(set(dict_obj[key]))
    return dict_obj


def df_mean_df_std_from_model(model, list_clusters):
    mean_list = list()
    std_list = list()
    for n in list_clusters:
        mean_list.append(model.means_[n])
        std_list.append(np.sqrt(np.diag(model.covars_[n])))

    df_mean = pd.DataFrame(mean_list, index=list_clusters)
    df_std = pd.DataFrame(std_list, index=list_clusters)
    return df_mean, df_std


def get_expected_profiles_from(df_mean, df_with, n_expected_clusters):
    n_real_time = len(df_with.index)
    df_mean_pt = df_mean.T.iloc[:n_real_time]
    df_error = df_mean_pt.sub(df_with.T, axis=0)
    df_error = df_error * df_error
    result = df_error.sum()
    rs = result.nsmallest(n_expected_clusters)
    return rs


def define_profile_area_from(selected_clusters_list, df_mean, df_std):
    mask = df_mean.index.isin(selected_clusters_list)
    df_area = df_mean[mask]
    df_profile = pd.DataFrame()
    df_profile['min'] = df_area.min()
    df_profile['max'] = df_area.max()
    df_profile["expected"] = df_area.mean()
    return df_profile, df_std[mask]


def obtain_expected_area(model_path, data_path, tag_name):

    """ Getting the HMM model, df_x (samples), df_y (labels) """
    model, df_x, df_y = hmm_u.get_model_dfx_dfy(model_path, data_path, filter_values=True, verbose=False)
    n_comp = model.n_components

    """ Grouping profiles according the name of the day
        Ex: {"Monday": [23,12,...], "sp_Monday":[12,14,...], 
             "Wednesday": [2,10,...], "sp_Wednesday":[13,44,...],  
             "special":[13,25, ...] }
    """
    dict_cl_week = day_cluster_matrix(model, df_y)

    pi_svr = pi.PIserver()
    pt = pi.PI_point(pi_svr, tag_name)