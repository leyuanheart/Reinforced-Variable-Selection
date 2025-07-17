import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import random
import copy
from tqdm import tqdm

# import sklearn
from sklearn.linear_model import LinearRegression, LassoLarsIC, LassoCV, Ridge, LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from scipy.stats import multivariate_normal, bernoulli

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Bernoulli
# from torchsummary import summary

import multiprocessing as mp


# ===================== Binary classification =====================================================
def generate_data(m=100, n=20, signal=1, sigma=1, num_support=8, seed=1):
    "Generates data matrix X and observations Y."
    np.random.seed(seed)
    # beta_star = np.random.randn(n)
    # beta_star[num_support:] = 0
        
    beta_star = np.zeros(n)
    beta_star[:num_support] = signal
    X = np.random.randn(m,n)
    logits = X.dot(beta_star) # + np.random.normal(0, sigma, size=m)   # 误差加在这结果会好一些
    p = 1 / (1 + np.exp(-logits))                                    # 误差如果加在这，就会出现一些问题，p和x的关系就会被破坏
    Y = (p > 0.5).astype(np.int)
    return X, Y, beta_star, np.diag(np.ones(n))


# ===================== Multi-class classification =====================================================
# def generate_data(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, seed=1):
#     X, Y = make_classification(n_samples=n_samples, 
#                                n_features=n_features, 
#                                n_informative=n_informative, 
#                                n_redundant=n_redundant,
#                                n_repeated = n_repeated,
#                                n_classes=n_classes, 
#                                shuffle=False, random_state=seed)
#     return X, Y


# ==============================================================================================



def compute_reward(X_train, Y_train, X_test, Y_test, actions, hiddens=(128, ), num_iter=500, lr=1e-3, batch_size='auto', dictionary=dict()):
    reward_list = []
    for i, action in enumerate(actions):
        
        idx = np.where(action == 1)[0]
        
        if tuple(idx) in dictionary:
            reward_list.append(dictionary[tuple(idx)])
        else:
            X_select = X_train[:, idx]        
            reg_clf = MLPRegressor(hidden_layer_sizes=hiddens, random_state=i, learning_rate='adaptive', batch_size=batch_size,
                                     learning_rate_init=lr, max_iter=num_iter, tol=1e-3, alpha=0.01, early_stopping=True)
#             reg_clf = LinearRegression(fit_intercept=False)
            # reg_clf = Ridge(alpha=0.1)
            # reg_clf = RandomForestRegressor(n_estimators=50, max_depth=5)
            # reg_clf = ExtraTreesRegressor(n_estimators=50, max_depth=5)
#             reg_clf = MLPClassifier(hidden_layer_sizes=hiddens, random_state=i, learning_rate='adaptive', batch_size=batch_size,
#                                      learning_rate_init=lr, max_iter=num_iter, tol=1e-3, alpha=0.01, early_stopping=True)
            # reg_clf = RandomClassifier(n_estimators=50, max_depth=5)
            # reg_clf = ExtraTreesClassifier(n_estimators=50, max_depth=5)
            reg_clf.fit(X_select, Y_train)
            X_select = X_test[:, idx] 
            score = reg_clf.score(X_select, Y_test)
            # mse = np.mean((Y_test - regressor.predict(X_select))**2)
            dictionary[tuple(idx)] = score
            reward_list.append(score)
        
    return np.array(reward_list)


# ========================= training steps ====================================================
# training steps
m = 100
n = 24
sigma = 0.5
num_support = 8
signal = 1
y_true = np.zeros(n, dtype=np.int)
y_true[:num_support] = 1


# # For multi-class classification
# n_informative = 8
# n_redundant = 0
# n_repeated = 0
# n_classes = 5
# num_support = n_informative + n_redundant

batch_size = 64

def run(seed):
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X, Y, beta_star, cov = generate_data(m, n, signal, sigma, num_support, seed=seed)   
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    
    
    r_list = []
    dictionary = dict()
    theta = np.zeros(n) + 0.5
    p_list = []
    w_norm = []
    
    for step in range(250):
        # print('step: ', step)
        
        p_list.append(theta)
        actions = np.zeros((batch_size, n))
        p = torch.from_numpy(theta)
        for i in range(batch_size):
            action = torch.bernoulli(p)
            if action.sum() == 0:
                idx = np.random.randint(0, n, int(n/3))
                action[idx] = 1
            actions[i, :] = action.numpy()


        rewards = compute_reward(x_train, y_train, x_test, y_test, actions, hiddens=(128, ), num_iter=1000, lr=1e-2, batch_size=batch_size, dictionary=dictionary)
        r_list.append(rewards.mean())
#         print(f'average reward: {rewards.mean()}')
    #     rewards = torch.tensor(rewards, dtype=torch.float32)

        # r_baseline = 0.95 * r_baseline + 0.05 * rewards.mean()

        # sampled natural policy gradient
        log_pi_grad = actions / theta - (1 - actions)/(1 - theta)


        reg = Ridge(alpha=0.1)
        reg.fit(log_pi_grad, rewards)
        w = reg.coef_

        w_norm.append(np.linalg.norm(w))

        theta = theta + 1 * w    
        theta = np.clip(theta, 0.02, 0.98)


    #     if step > 6:
    #         if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
    #             & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
    #             print(f'converge at step {step}')
    #             break

        if np.linalg.norm(theta - p_list[-1]) < 1e-3:
            print(f'converge at step {step}')
            break
            
    
    tmp = sorted(dictionary.items(), key=lambda x: x[1])
    s = set(range(n))
    for item in tmp[:10]:
        s = s & set(item[0])
    # print(s)
    
                        
    y_pred_rl1 = theta
    y_pred_rl2 = np.where([i in s for i in range(n)], 1, 0)
    
    
    lasso_bic = LassoLarsIC(criterion='bic', fit_intercept=False, normalize=False)
    lasso_bic.fit(X, Y)
    y_pred_bic = np.where(lasso_bic.coef_ != 0, 1, 0)
    lasso_aic = LassoLarsIC(criterion='aic', fit_intercept=False, normalize=False)
    lasso_aic.fit(X, Y)
    y_pred_aic = np.where(lasso_aic.coef_ != 0, 1, 0)
    rf = RandomForestRegressor(max_depth=5, random_state=seed)    
    rf.fit(X, Y)
    sfm = SelectFromModel(rf, prefit=True)
    y_pred_sfm = np.where(sfm.get_support() != 0, 1, 0)
    
    dat = np.vstack((y_pred_rl1, y_pred_rl2, y_pred_aic, y_pred_bic, y_pred_sfm))
    
    # cm1 = confusion_matrix(y_true, y_pred_rl1)
    # cm2 = confusion_matrix(y_true, y_pred_rl2)
    # cm_bic = confusion_matrix(y_true, y_pred_bic)
    # cm_aic = confusion_matrix(y_true, y_pred_aic)
    # # return cm1, cm2, cm_bic, cm_aic
    
    # dat = pd.DataFrame(np.zeros((3, 4)), index=['precision', 'specificity', 'recall'])
    #                     # columns=['cm1', 'cm2', 'bic', 'aic'])
    
    # for i, cm in enumerate([cm1, cm2, cm_bic, cm_aic]):
    #     tn, fp, fn, tp = cm.ravel()
    #     dat.loc['precision', i] = tp/(tp+fn)
    #     dat.loc['specificity', i] = tn/(tn+fp)
    #     dat.loc['recall', i] = 0 if tp + fp == 0 else tp/(tp+fp)
        
    # dat.columns = ['cm1', 'cm2', 'bic', 'aic']
    
    
#     regr = LogisticRegression(penalty='l2', fit_intercept=False, max_iter=1e6)
#     regr.fit(X, Y)
#     sfm = SelectFromModel(regr, prefit=True)
#     y_pred_log = np.where(sfm.get_support(), 1, 0)
    
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return dat


if __name__ == '__main__':   
    # results = []
    # for sd in tqdm(range(20)):
    #     results.append(run(sd))

    # print("CPU的核数为：{}".format(mp.cpu_count()))
    start = time.time()
    pool = mp.Pool(4)
    dats = pool.map(run, range(50))
    pool.close()
    end = time.time()
    print(datetime.timedelta(seconds = end - start))
    
    
    dats = np.array([dat for dat in dats])

    np.save('./results/m100_n24_logistic_reg_coef1_actor_step200_1e-3_predictor_step800_lr1e-2_last10_predbz64_1.5h', dats)