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
from my_lib.holidays import holidays as hl

from plotly import tools #to do subplots
import plotly.offline as py
import cufflinks as cf
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
import pylab as pl
from IPython.display import display
py.init_notebook_mode(connected=False) # run at the start of every ipython notebook to use plotly.offline

max_alpha = 2
min_step = 0.1
alpha_values = np.arange(-1.5, max_alpha, min_step)
n_allowed_consecutive_violations = 3
n_profiles_to_see = 5

def day_cluster_matrix(hmm_model, df_y):
    n_comp = hmm_model.n_components
    dict_cl_week = dict()
    sp = min(0.005 * len(df_y.index), 3)

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
            dict_cl_week = append_in_list(dict_cl_week, "atypical", n)

    dict_cl_week["holidays"] = hl.get_holidays_dates()
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


def df_mean_df_std_from_holidays(df_x, list_holidays):
    mask = df_x.index.isin(list_holidays)
    df_mean = df_x[mask]
    std_list = list(df_mean.std()*0.10)
    std_list = [std_list for i in range(len(df_mean.index))]
    df_std = pd.DataFrame(std_list, index=df_mean.index)
    return df_mean, df_std


def get_expected_profiles_from(df_mean, df_with, n_expected_clusters):
    n_real_time = len(df_with.index)
    df_mean_pt = df_mean.T.iloc[:n_real_time]
    df_error = df_mean_pt.sub(df_with.T, axis=0)
    df_error = df_error * df_error
    result = df_error.sum().pow(1/2)
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


def obtain_expected_area(model_path, data_path, tag_name, str_time_ini, str_time_end):

    """ Getting the HMM model, df_x (samples), df_y (labels) """
    model, df_x, df_y = hmm_u.get_model_dfx_dfy(model_path, data_path, filter_values=True, verbose=False)

    """ Grouping profiles according the name of the day
        Ex: {"Monday": [23,12,...], "sp_Monday":[12,14,...], 
             "Wednesday": [2,10,...], "sp_Wednesday":[13,44,...],  
             "atypical":[13,25, ...] }
    """
    dict_cl_week = day_cluster_matrix(model, df_y)

    """ Getting information from the PIserver """
    pi_svr = pi.PIserver()
    pt = pi.PI_point(pi_svr, tag_name)
    time_range = pi_svr.time_range(str_time_ini, str_time_end)
    span = pi_svr.span("30m")
    df_int = pt.interpolated(time_range, span)
    d_week = df_int.index[0].weekday_name

    """ Find profile according to the family of profiles: """
    profile_families = [d_week, "sp_" + d_week, "atypical", "holidays"]

    min_error = np.inf
    family_error_dict = dict()
    result = dict()
    for family in profile_families:
        list_clusters = dict_cl_week[family]

        if family != "holidays":
            """ Get the mean and std of the model according the list of clusters"""
            df_model_mean, df_model_std = df_mean_df_std_from_model(model, list_clusters)

        else:
            """ Get the mean and std of from holidays"""
            df_model_mean, df_model_std = df_mean_df_std_from_holidays(df_x, list_clusters)

        """ Setting the today timestamps for ploting"""
        n_columns = pd.date_range(df_int.index[0], df_int.index[0] + pd.Timedelta('23 H 30 m'), freq='30T')
        df_model_mean.columns = n_columns
        df_model_std.columns = n_columns

        """ Find the n best profiles """
        family_error = np.inf
        for n_profiles in range(1, n_profiles_to_see+1):

            rs = get_expected_profiles_from(df_model_mean, df_with=df_int[tag_name],
                                            n_expected_clusters=n_profiles)

            # Define the n best expected profiles and the shape of the expected profile
            expected_profiles = list(rs.index.values)
            df_profile, df_profile_std = define_profile_area_from(expected_profiles, df_model_mean, df_model_std)
            df_std = df_profile_std.mean()

            # Find the expected area and adjust the band to do not make n violations
            df_expected_area, alpha_min_max = adjust_expect_band(df_int[tag_name], df_profile, df_std)
            error = estimate_error(df_int[tag_name], df_expected_area["expected"])

            if error < min_error and abs(alpha_min_max[0]) < max_alpha and abs(alpha_min_max[1]) < max_alpha :
                min_error = error
                result["df_expected_area"] = df_expected_area
                result["family"] = family
                result["n_profiles"] = n_profiles
                result["expected_profiles"] = expected_profiles
                result["alpha_values"] = alpha_min_max
                result["min_error"] = min_error
                alpha_final = max(abs(alpha_min_max[0]), abs(alpha_min_max[1]))
                result["df_std"] = df_std*alpha_final

            if error < family_error and abs(alpha_min_max[0]) < max_alpha and abs(alpha_min_max[1]) < max_alpha :
                family_error = error
                family_error_dict[family] = error
                result["family_error"] = family_error_dict

    return result


def adjust_expect_band(df_int, df_profile, df_std):

    df_aux = df_int.dropna()
    df_size = min(len(df_aux.index),len(df_profile.index))
    # Define the expected area:
    df_area = pd.DataFrame(index=df_profile.index)
    alpha_max_lim, alpha_min_lim = 0, 0

    # Adjusting the expected band by changing the alpha values:
    for alpha in alpha_values:
        df_area["min"] = df_profile['min'] - alpha * df_std.values
        # print(len(df_int.index))
        check_list = list(df_int[:df_size] < df_area["min"][:df_size])
        alpha_min_lim = alpha
        if not there_is_n_consecutive_violations(check_list, n_allowed_consecutive_violations):
            break

    for alpha in alpha_values:
        df_area["max"] = df_profile['max'] + alpha * df_std.values
        check_list = list(df_int[:df_size] > df_area["max"][:df_size])
        alpha_max_lim = alpha
        if not there_is_n_consecutive_violations(check_list, n_allowed_consecutive_violations):
            break

    alpha_min_lim += min_step
    alpha_max_lim += min_step

    df_area["max"] = df_profile['max'] + alpha_max_lim * df_std.values
    df_area["min"] = df_profile['min'] - alpha_min_lim * df_std.values

    df_area["expected"] = (df_area["max"] + df_area["min"]) / 2
    df_area["real time"] = df_int
    return df_area, [alpha_min_lim, alpha_max_lim]


# mean absolute percentage error (MAPE)
def estimate_error(df_real, df_predicted):
    # same size for two dataFrames
    n_real_time = len(df_real.index)
    df_predicted = df_predicted.iloc[:n_real_time]

    # Normalize values [0, 1]
    # max_value = max(df_real.max(), df_predicted.max())
    # min_value = min(df_real.min(), df_predicted.min())
    # range_value = max_value - min_value
    # df_real = (df_real - min_value) / range_value
    # df_predicted = (df_predicted - min_value) / range_value

    df_error = df_real.sub(df_predicted.T, axis=0).abs()
    df_error = df_error / df_real
    df_error = df_error.sum() / n_real_time

    return df_error*100


def there_is_n_consecutive_violations(check_list, n_violations):
    violations = ["True"]*n_violations
    str_violations = ", ".join(violations)
    # print(str(check_list).find(str(str_violations)))
    # print(check_list[-1])

    if check_list[-1]:
        return True
    if str(check_list).find(str(str_violations)) >= 0:
        return True
    else:
        return False


"""
    GRAPHICAL PART OF THIS MODULE
"""

layout = go.Layout(
    autosize=False,
    width=700,
    height=400,
    margin=go.Margin(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=0
    )
    # paper_bgcolor='#7f7f7f',
    # plot_bgcolor='#c7c7c7'
)


def traces_expected_area_and_real_time(df_expected_area):
    traces = list()
    colors = {'min': 'green', 'max': 'green', 'real time': 'red', 'expected': 'blue'}
    width = {'min': 1, 'max': 1, 'real time': 4, 'expected': 2}
    dash = {'min': 'dot', 'max': 'dot', 'real time': None, 'expected': 'dashdot'}
    fill = {'min': None, 'max': 'tonexty', 'real time': None, 'expected': None}

    for column in df_expected_area.columns:
        trace = go.Scatter(
            x=df_expected_area.index,
            y=df_expected_area[column],
            name=column,
            mode='line',
            fill=fill[column],
            line=dict(
                width=width[column],
                color=colors[column],
                dash=dash[column]
            )
        )

        traces.append(trace)
    return traces


def trace_df_std(df_std):
    trace = go.Scatter(
        x=df_std.index,
        y=df_std,
        name="standard deviation",
        mode='line',
        fill='tozeroy',
        xaxis='x',
        yaxis='y2',
        line=dict(
            width=1,
            color='orange',
            shape='hv'
        )

    )
    return trace