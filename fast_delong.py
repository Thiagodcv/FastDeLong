"""
The fast_delong module. This module provides the implementation for the two algorithms proposed in
"Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver 
Operating Characterisitic Curves" by Xu Sun and Weichao Xu. 

Algorithm 1: Procedure of Calculating Mid-ranks
    A fast algorithm for calculating the mid-ranks of a sequence.

Algorithm 2: Improved DeLong's Algorithm
    Fast implementation of DeLong's algorithm based on thereom 1 in the paper.

---------------------------------- Naming Conventions ----------------------------------
In this documentation, we go by the naming conventions of the paper. Let k be the total number 
of classifiers, and let r be a particular classifier with 1 <= r <= k.

The m predicted probabilities made by model r on AD-positive patients are {X_1^(r), ..., X_m^(r)}.

The n predicted probabilities made by model r on control patients are {Y_1^(r), ..., Y_n^(r)}.

When concatenating sequences {X_1^(r), ..., X_m^(r)} and {Y_1^(r), ..., Y_n^(r)} we are left 
with {Z_1^{r}, ..., Z_N^(r)}, where N = m + n. 
"""
import numpy as np 
import pandas as pd 
from scipy.stats import norm
import os

FILENAMES = ['unedited_data.csv', 'nodis_data.csv']
MODEL = 'LogReg'
PATH = '/home/thiago/FastDeLong/data'

def convert_canary_data():
    """
    Converts CANARY prediction csv files into a dataframe with columns
    "ground_truth", "pred_1", "pred_2", ... "pred_i" for k models used. 

    Returns
    -------
    (mxk)  ndarray
        Columns are the predicted probabilities output from k models
        for all AD-positive subjects

    (nxk)  ndarray
        Columns are the predicted probabilities output from k models
        for all control subjects

    """
    df = pd.read_csv(os.path.join(PATH, FILENAMES[0]))
    df = df.loc[df['model']==MODEL, ['PID', 'prob_1']]
    df.rename(columns={'PID':'ground_truth'}, inplace=True)

    for i in range(1, len(FILENAMES)):
        df_next = pd.read_csv(os.path.join(PATH, FILENAMES[i]))
        df_next = df_next.loc[df_next['model']==MODEL, 'prob_1']
        df['prob_{}'.format(i+1)] = df_next
    
    for i in range(len(df)):
        df.iloc[i, 0] = 0 if df.iloc[i, 0].startswith('H') else 1

    positive_samples = df[df['ground_truth']==1].iloc[:,1:]
    negative_samples = df[df['ground_truth']==0].iloc[:,1:]

    return positive_samples.to_numpy(), negative_samples.to_numpy()


def calculate_midranks(seq):
    """
    Calculates the mid-ranks of a sequence. So if sequence Z_1, ..., Z_M is input,
    outputs associated mid-ranks T_z(Z_1), ..., T_z(Z_M).

    We have that ``M`` is the length of the input sequence, ``I`` is a list of indices, where 
    I[j] = i implies that the jth entry of ``seq`` should have an index i were ``seq`` to be sorted
    in ascending order. ``W`` is simply just ``seq`` but in ascending order. T[k] = c implies that
    the kth element of ``W`` has a midrank of c, and T_z[k] = c implies that the kth element of 
    ``seq`` has a midrank of c when ``seq`` is sorted into ``W``.

    The indices ''i'' and ''j'' iterate through the list to find the beginning and end of subsequences of
    identical numbers within ''W''. If a subsequence is of length 1 (a unique number within the set of ''W''), then
    ''a'' = ''b'' = index of the unique number. If a subsequence is of length greater than 1, then 
    ''a'' = index of beginning of subsequence in ''W'' < ''b'' = index of end of subsequence in ''W'', and the midrank
    of this subsequence is equal to (a+b)/2 by equation 11 in the paper cited above. The indices ''i'', ''j'', ''a'',
    and ''b'' are all 1-indexed, so when accessing memory in lists, we must subtract 1 from the index (i.e. W[a-1]).

    Parameters
    ----------
    seq: (Mx1) ndarray
    
    Returns
    -------
    ndarray
        of size (Mx1)
    """

    M = len(seq)
    I = np.argsort(seq, kind='quicksort')
    W = [seq[i] for i in I] + [np.max(seq) + 1]
    T = np.zeros(M)
    T_z = np.zeros(M)
    
    i=1
    while i <= M:
        a = i 
        j = a 
        while W[j-1] == W[a-1]:
            j = j+1
            
        b = j-1 
        for k in range(a, b+1):
            T[k-1] = (a+b)/2

        i = b+1

    for i in range(0, M):
        c = I[i]
        T_z[c] = T[i] 

    return T_z
        

def fast_delong(X, Y):
    """
    Calculates and returns estimator of AUC ``theta_hat``, and 
    variance-covariance matrix estimator ``S`` for ``theta_hat``.

    Parameters
    ----------
    X: (mxk) ndarray
        An array where X[i,:] = the probability estimates from all k models for AD-positive subject i,
        and X[:,j] = the probability estimates for all AD-positive subjects from model j
    Y: (nxk) ndarray
        An array where Y[i,:] = the probability estimates from all k models for control subject i,
        and Y[:,j] = the probability estmimates for all control subjects from model j

    Returns
    -------
    (kxk) ndarray
        The variance-covariance estimator ``S``

    (kx1) ndarray
        The AUC estimator ``theta_hat``
    """

    m = X.shape[0]
    n = Y.shape[0]
    k = X.shape[1]

    Z = np.concatenate((X, Y), axis=0)

    T_z = np.zeros((m+n, k))
    T_x = np.zeros((m, k))
    T_y = np.zeros((n, k))
    
    for r in range(0, k):
        T_z[:, r] = calculate_midranks(Z[:, r])
        T_x[:, r] = calculate_midranks(X[:, r])
        T_y[:, r] = calculate_midranks(Y[:, r])

    V_10 = (T_z[:m, :] - T_x)/n
    theta_hat = np.sum(T_z[:m, :], axis=0)
    theta_hat = theta_hat/(m*n) - (m+1)/(2*n)
    V_01 = 1 - (T_z[m:, :] - T_y)/m
    
    S_10 = np.zeros((k, k))
    S_01 = np.zeros((k, k))
    S = np.zeros((k, k))

    for r in range(0, k):
        for s in range(0, k):

            for i in range(0, m):
                S_10[r, s] = S_10[r, s] + (V_10[i, r] - theta_hat[r])*(V_10[i, s] - theta_hat[s])
            
            for j in range(0, n):
                S_01[r, s] = S_01[r, s] + (V_01[j, r] - theta_hat[r])*(V_01[j, s] - theta_hat[s])
            
            S[r, s] = S_10[r, s]/(m-1)/m + S_01[r, s]/(n-1)/n

    return S, theta_hat


def get_test_statistic(S, theta_hat):
    """
    Returns the test statistic of the Delong test. Note that the null hypothesis
    is set to L * theta = 0, where L = [1, -1]. This is equivalent to 
    theta[0] - theta[1] = 0.

    Parameters
    ----------
    S: (kxk) ndarray
        The variance-covariance estimator ``S``

    theta_hat: (kx1) ndarray
        The AUC estimator ``theta_hat``

    Returns
    -------
    float
        The test statistic
    """

    L = np.array([1, -1])
    t = (L.T @ theta_hat)/(np.sqrt(L.T @ S @ L)) # This is standard normal
    return t


def delong_test(test_type, alpha):
    """
    Performs the DeLong test on two different models.

    To specifiy the classification algorithm, the 2 files containing the 2 sets of classification results,
    and the path to the files, modify the global variables ``MODEL``,  ``FILENAMES``, and ``PATH`` respectively.

    Parameters
    ----------
    test_type: str
        Can be set to 'lower', 'upper', or 'two-tailed' for lower-tail, upper-tail, and two-tailed
        hypothesis testing respectively

    alpha: float
        The significance value, or the probability of rejecting the null hypothesis when it is true
        
    Returns
    -------
    dict
        Of the form {'T':t, 'p':p, 'lower_CI':lci, 'upper_CI':uci} where t is the test statistic, p is the p-value,
        lci is the lower confidence interval, and uci is the upper confidence interval
    """

    X, Y = convert_canary_data()
    S, theta_hat = fast_delong(X, Y)
    t = get_test_statistic(S, theta_hat)

    # additional parameters for calculating confidence intervals.
    stdev = np.sqrt(S[0,0] + S[1,1] + 2*S[0,1])
    theta_dif = theta_hat[1] - theta_hat[0]

    if test_type == 'lower':
        p = norm.cdf(t)
        lci = theta_dif + norm.ppf(alpha)*stdev
        uci = None
    elif test_type == 'upper':
        p = 1 - norm.cdf(t)
        lci = None
        uci = theta_dif - norm.ppf(alpha)*stdev
    elif test_type == 'two-tailed':
        p = 2*min(norm.cdf(t), 1 - norm.cdf(t))
        lci = theta_dif + norm.ppf(alpha/2)*stdev
        uci = theta_dif - norm.ppf(alpha/2)*stdev
    else:
        print('Wrong test_type!')

    return {'T':t, 'p':p, 'lower_CI':lci, 'Upper_CI':uci}
