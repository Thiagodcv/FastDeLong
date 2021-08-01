"""
The test module -- tests functions from fast_delong.py.
"""
from fast_delong import calculate_midranks, fast_delong, get_test_statistic
import numpy as np
from math import fabs

def test_calculate_midranks():
    assert list(calculate_midranks([1.3, 1.7, 1.7, 2.5])) == [1, 2.5, 2.5, 4]
    assert list(calculate_midranks([1.7, 1.3, 1.7, 2.5])) == [2.5, 1, 2.5, 4]
    assert list(calculate_midranks([4, 1, 0, 9, 22, 0])) == [4, 3, 1.5, 5, 6, 1.5]


def test_fast_delong():
    AD_1_preds = [0.6, 0.2]
    AD_2_preds = [0.7, 0.7]
    AD_3_preds = [0.8, 0.9]
    X = np.array([AD_1_preds, AD_2_preds, AD_3_preds])

    control_1_preds = [0.1, 0.3]
    control_2_preds = [0.2, 0.6]
    Y = np.array([control_1_preds, control_2_preds])

    S, theta_hat = fast_delong(X, Y)

    #print('S: ', S)
    #print('theta_hat: ', theta_hat)

    S_expected = np.array([[0,0], 
                           [0, 1/9]])
    theta_hat_expected = np.array([1, 2/3])
    
    assert S.shape[0] == S.shape[1]
    assert S.shape[0] == S_expected.shape[0] and S.shape[1] == S_expected.shape[1]
    assert theta_hat.shape[0] == theta_hat_expected.shape[0]
    
    k = S_expected.shape[0]

    for i in range(k):
        for j in range(k):
            assert fabs(S[i,j] - S_expected[i,j]) < 0.0001

    for c in range(theta_hat_expected.shape[0]):
        assert fabs(theta_hat[c] - theta_hat_expected[c]) < 0.0001

    
def test_get_test_statistic():
    AD_1_preds = [0.6, 0.2]
    AD_2_preds = [0.7, 0.7]
    AD_3_preds = [0.8, 0.9]
    X = np.array([AD_1_preds, AD_2_preds, AD_3_preds])

    control_1_preds = [0.1, 0.3]
    control_2_preds = [0.2, 0.6]
    Y = np.array([control_1_preds, control_2_preds])

    S, theta_hat = fast_delong(X, Y)

    t = get_test_statistic(S, theta_hat)
    
    assert fabs(t - 1) < 0.00001
