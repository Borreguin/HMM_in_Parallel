import warnings

import ipyparallel as ipp
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.externals import joblib


def h(x):
    return -3 * pow(x, 3) + 2 * pow(x, 2) + 5 * x

def get_ipp_client(profile='default'):
    rc = None
    try:
        rc = ipp.Client(profile=profile)
        print("Engines running for this client: {0}".format(rc.ids))
    except:
        print("Make sure you are running engines by the command: \n $ipcluster start --profile=default -n 4")
    return rc


def pivot_DF_using_dates_and_hours(df):
    df = df[~df.index.duplicated(keep='first')]
    """ Allow to pivot the dataframe using dates and hours"""
    df["hour"] = [x.time() for x in df.index]
    df['date'] = [x._date_repr for x in df.index]
    # transform series in a table for hour and dates
    try:
        df = df.pivot(index='date', columns='hour')
        #df.fillna(method='pad', inplace=True)
        df.dropna(inplace=True)
    except:
        print('No possible convertion, in format: index-> date, columns-> hours')
        df = pd.DataFrame()

    return df


def select_best_HMM(training_set, validating_set, nComp_list, seed=777):

    """ client is an instance of ipp.Client:
        df_dataSet is an instance of pd.DataFrame
    """
    best_score, best_log_prob = 0, -np.inf  # best_score in [0 to 1] and best_log_prob > -np.inf
    best_model, log_register_list = None, list()
    np.random.seed(seed)  # different random seed
    for n_component in nComp_list:
        try:
            model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=100).fit(training_set)
            assert isinstance(model,GaussianHMM)
            score, log_prob = score_model(validating_set, model)
            log_register_list.append((n_component, round(score,5), round(log_prob,5)))

            if score > best_score and log_prob > best_log_prob:
                best_score = score
                best_model = model
                best_log_prob = log_prob
        except:
            return None, None
    #return best_model, score_list, best_log_prob
    return best_model, log_register_list


def score_model(validating_set, model):
    r, n = 0, len(validating_set)

    try:
        score_samples = model.predict_proba(validating_set)
        log_prob = model.score(validating_set)
        for sample_score in score_samples:
            max_prob = max(sample_score)
            r += max_prob

        score = (r / n)
    except:
        return 0, -np.inf

    return score, log_prob


def select_best_model_from_list(best_model_list, validating_set, verbose=True):

    #warnings.warn("deprecated", DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    best_score, best_log_prob = 0, -np.inf
    best_model, log_register = None, list()
    last_register = list()
    err = 0
    for item_model in best_model_list:
        model = item_model['model']
        log_register += item_model['log_register']
        if model is not None:
            assert isinstance(model, GaussianHMM)
            score, log_prob = score_model(validating_set, model)
            last_register.append((model.n_components, round(score,5), round(log_prob,5)))
            if score > best_score and log_prob > best_log_prob:
                best_score = score
                best_model = model
                best_log_prob = log_prob
        else:
            err += 1
    if verbose:
        # print('Trained models: {0}'.format(last_register))
        print('\tBest model: \t\t\t\tnComp={0}, score={1:3.4f}, log_prob={2:5.2f}'.format(
            best_model.n_components, best_score, best_log_prob) )
        if err > 0:
            print("\tThere is {0} errors related to trained models".format(err))

    return best_model, log_register + last_register


def ordered_hmm_model(model, method='average', metric='euclidean'):
    """
    From a trained model, creates a new model that reorder the means of the model according
    to the hierarchical clustering HC
    :param model:  a trained Hidden Markov Model
    :param method: Available methods: 'average', 'single', 'complete', 'median', 'ward', 'weighted'
    :param metric: Available metrics: 'euclidean', 'minkowski', 'cityblock', 'sqeuclidean'
    :return: A ordered hmm model
    """
    from scipy.cluster.hierarchy import linkage
    import copy
    from hmmlearn.hmm import GaussianHMM
    ordered_model = copy.deepcopy(model)

    #try:
    # assert isinstance(model,GaussianHMM)

    """ Z_f contains the distance matrix of the means of the model """
    Z_f = linkage(model.means_, method=method, metric=metric)

    """ Create a new order for the means of the model according to the hierarchical clustering """

    n_comp, new_order = model.n_components, list()
    for idx, idy, d, c in Z_f:
        if idx < n_comp:
            new_order.append(int(idx))
        if idy < n_comp:
            new_order.append(int(idy))


    """ Ordering the means and covars according to 'new_order': """
    # The use of model._covars_ is exceptional, usually it should be "model.covars_"

    old_means, old_covars = model.means_, model._covars_
    new_means, new_covars = np.zeros_like(old_means), np.zeros_like(old_covars)
    for idx, re_idx in zip(list(range(n_comp)), new_order):
        new_means[idx] = old_means[re_idx]
        new_covars[idx] = old_covars[re_idx]

    """ Ordering transition matrix B and  start probability \pi """
    old_transmat, new_transmat = model.transmat_, np.zeros_like(model.transmat_)
    n = old_transmat.shape[0]
    for x in list(range(n)):
        for y in list(range(n)):
            new_transmat[y, x] = old_transmat[new_order[y], new_order[x]]

    start_p = np.array([1 / n_comp for i in range(n_comp)])

    """ Setting the new ordered model """
    ordered_model.startprob_ = start_p
    ordered_model.transmat_ = new_transmat
    ordered_model.means_ = new_means
    ordered_model.covars_ = new_covars

    return ordered_model
    #except:
    #    return model



def save_model_and_log(model, log_register, model_path, log_path, file_name):
    file1 = model_path + file_name
    file2 = log_path + "log_" + file_name

    try:
        joblib.dump(model, filename=file1, compress=3, protocol=2)
        joblib.dump(log_register, filename=file2, compress=3, protocol=2)

    except FileNotFoundError:
        file1 = "./" + file_name
        file2 = "./" + "log_" + file_name
        joblib.dump(model, filename=file1 , compress=3, protocol=2)
        joblib.dump(model, filename=file2 , compress=3, protocol=2)

    print('\tBest model saved in: \t\t\t', file1)
    print('\tLog register in: \t\t\t', file2)


# Shared memory is most costly instead break down the DataFrame
# mgr = Manager()
# ns = mgr.Namespace()
# ns.df = pd.DataFrame(list(range(1, 500000)))
# ns.dt = dataSet