import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import os
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score



paths = os.listdir('.')
paths = np.array(paths)

# 0  'cnae-9_cv_npg.npy',
# 1  'cnae-9_scale_cv_npg.npy',
# 2  'crime_cv_npg.npy',
# 3  'figures_tables.py',
# 4  'm100_n1000_linear_correlated_var0.9_signal1_sigma0.5_step1500_batch64_linearfunc_0.9.npy',
# 5  'm100_n300_linear_signal1_sigma0.5_step200_batch64_linearfunc_0.9.npy',
# 6  'm100_n50_linear_correlated_var0.9_signal1_sigma0.5_step500_batch64_linearfunc_0.7.npy',
# 7  'm100_n50_linear_signal1_sigma0.5_step200_batch64_linearfunc_0.9.npy',
# 8  'm200_n1000_linear_correlated_var0.9_signal1_sigma0.5_step1500_batch64_ridgefunc_0.7.npy',
# 9  'm200_n1000_linear_signal1_sigma0.5_step1000_batch64_ridgefunc_0.9.npy',
# 10 'm200_n300_logistic_sigma0.5_step450_batch64_mlp_reward_0.75.npy',
# 11 'm200_n30_intersection_signal1_sigma1_step100_batch64_mlp_reward_4.8h.npy',
# 12 'm200_n50_binary_classification_signal1_sigma0.5_step150_batch64_mlp_reward_14.5h.npy',
# 13 'm200_n50_hierarchical_sigma0.5_step150_batch64_mlp_reward.npy',
# 14 'm200_n50_intersection_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 15 'm200_n50_linear_correlated_var0.9_signal1_sigma0.5_step750_batch64_linearfunc_0.75.npy',
# 16 'm200_n50_linear_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 17 'm200_n50_make_classification_step200_batch64_mlp_reward_19.5h.npy',
# 18 'm300_n50_linear_correlated_var0.9_signal1_sigma0.5_step1000_batch64_linearfunc_0.8.npy',
# 19 'm300_n50_linear_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 20 'm400_n50_linear_correlated_var0.9_signal1_sigma0.5_step500_batch64_linearfunc_0.75.npy',
# 21 'm400_n50_linear_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 22 'madelon_cv_npg.npy',
# 23 'spam_cv_npg.npy'
# 24 'z-crime_cv_npg_new.npy'
# 25 'zz-cnae-9_cv_npg_new.npy'


'''
linear m=100, 200, 300, 400, n=50, [7, 16, 19, 21], threshold=0.9
linear m=100, n=300, [5], threshold=0.9
linear m=200, n=300, [9], threshold=0.9

linear correlated m=100, 200, 300, 400, n=50, [6, 15, 18, 20], threshold=0.75
linear correlated m=100,200, n=1000, [4, 8], threshold=[0.9, 0.7]


cross term m=200, n=30, [11], threshold=0.9
hierarchical m=200, n=50, [13], threshold=0.8
binary classification m=200, n=50, [12], threshold=0.9
multi-classification m=200, n=50, [17], threshold=0.9

cross term m=200, n=50, [14], threshold=0.9, reward=linear_func


spam [23]
crime(new) [2(24)]
madelon [22]
cnae-9(scale) [0(1)]

'''


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    # balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall =recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def regression_result(dats, threshold):
    _, _, n = dats.shape
    num_support = 8
    y_true = np.zeros(n, dtype=int)
    y_true[:num_support] = 1
    
    theta = dats[:,0,:]
    y_pred_rl1 = np.where(theta > threshold, 1, 0)
    
    y_pred_aic = np.zeros((50, n))
    nums = np.sum(dats[:, 1, :] != 0, axis=1)
    for i, num in enumerate(nums):
        idx = np.where(dats[i, 1, :] != 0)[0]
        y_pred_aic[i, idx] = 1
          
    y_pred_bic = np.zeros((50, n))
    nums = np.sum(dats[:, 2, :] != 0, axis=1)
    for i, num in enumerate(nums):
        idx = np.where(dats[i, 2, :] != 0)[0]
        y_pred_bic[i, idx] = 1
        
    y_pred_cv = np.zeros((50, n))
    nums = np.sum(dats[:, 3, :] != 0, axis=1)
    for i, num in enumerate(nums):
        idx = np.where(dats[i, 3, :] != 0)[0]
        y_pred_cv[i, idx] = 1  
    
    rf_means = np.mean(dats[:, 4, :], axis=1)
    y_pred_rf = np.zeros((50, n))
    for i, mean in enumerate(rf_means):
        y_pred_rf[i, :] =np.where(dats[i, 4, :] > mean, 1, 0)
    
    result = []
    for i, y_preds in enumerate([y_pred_rl1, y_pred_aic, y_pred_bic, y_pred_cv, y_pred_rf]):
        result.append([compute_metrics(y_true, y_pred) for y_pred  in y_preds])
    
    result = np.array(result)
    
    return result



def classification_result(dats, threshold):
    _, _, n = dats.shape
    num_support = 8
    y_true = np.zeros(n, dtype=int)
    y_true[:num_support] = 1
    
    theta = dats[:,0,:]
    y_pred_rl1 = np.where(theta > threshold, 1, 0)
    
    y_pred_cv = np.zeros((50, n))
    nums = np.sum(dats[:, 1, :] != 0, axis=1)
    for i, num in enumerate(nums):
        idx = np.where(dats[i, 1, :] != 0)[0]
        y_pred_cv[i, idx] = 1  
    
    
    means = np.mean(dats[:, 2, :], axis=1)
    y_pred_rf = np.zeros((50, n))
    for i, mean in enumerate(means):
        y_pred_rf[i, :] =np.where(dats[i, 2, :] > mean, 1, 0)
    
    result = []
    for i, y_preds in enumerate([y_pred_rl1, y_pred_cv, y_pred_rf]):
        result.append([compute_metrics(y_true, y_pred) for y_pred  in y_preds])
    
    result = np.array(result)
    
    return result



#========================== synthetic results ==========================

def linear_cases():
    pass
## ============ linear cases =========================
# urls = paths[[7, 16, 19, 21]]
# thresholds = [0.9, 0.9, 0.9, 0.9]
urls = paths[[6, 15, 18, 20]]
thresholds = [0.75, 0.75, 0.75, 0.75]
tmp = np.load(urls[0])

_, _, n = tmp.shape
num_support = 8
y_true = np.zeros(n, dtype=int)
y_true[:num_support] = 1


results = []
for url, threshold in zip(urls, thresholds):
    dats = np.load(url)
    results.append(regression_result(dats, threshold))
    
results = np.array(results)   # (n_samples, methods_types, runs, metrics)
    
means = results.mean(axis=2)   # (n_samples, method_types, metrics) 
stds = results.std(axis=2)

    
metric_names = ['Accuracy', 'Precision', 'Recall', r'$F_1$ score']
method_names = ['ACP', 'Lasso_AIC', 'Lasso_BIC', 'Lasso_CV', 'Random Forest']


def different_sample_sizes():    
    x_ticks = np.array([200, 400, 600, 800])
    width = 25
    fig = plt.figure() 
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        rects1 = ax.bar(x_ticks - 2*width, means[i, 0, :], width, yerr=stds[i, 0, :], label=method_names[0])
        rects2 = ax.bar(x_ticks - 1*width, means[i, 1, :], width, yerr=stds[i, 1, :], label=method_names[1])
        rects3 = ax.bar(x_ticks , means[i, 2, :], width, yerr=stds[i, 2, :], label=method_names[2])
        rects4 = ax.bar(x_ticks + 1*width, means[i, 3, :], width, yerr=stds[i, 3, :], label=method_names[3])
        rects5 = ax.bar(x_ticks + 2*width, means[i, 4, :], width, yerr=stds[i, 4, :], label=method_names[4])
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
        ax.set_ylim(0, 1)
        ax.set_title(f'Sample size: {(i+1)*100}')
    ax.legend(loc='best', bbox_to_anchor=(1.05,1.0))
    

def different_metrics():
    pass
x_ticks = np.array([200, 400, 600, 800])
width = 25
fig = plt.figure() 
fig.subplots_adjust(top=0.959,
                    bottom=0.057,
                    left=0.104,
                    right=0.882,
                    hspace=0.164,
                    wspace=0.252)
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    ax.errorbar(x_ticks - 2*width, means[:, 0, i], yerr=stds[:, 0, i], linestyle='--', fmt='.', label=method_names[0])
    ax.errorbar(x_ticks - 1*width, means[:, 1, i], yerr=stds[:, 1, i], linestyle='--', fmt='.', label=method_names[1])
    ax.errorbar(x_ticks, means[:, 2, i], yerr=stds[:, 2, i], linestyle='--', fmt='.', label=method_names[2])
    ax.errorbar(x_ticks + 1*width, means[:, 3, i], yerr=stds[:, 3, i], linestyle='--', fmt='.', label=method_names[3])
    ax.errorbar(x_ticks + 2*width, means[:, 4, i], yerr=stds[:, 4, i], linestyle='--', fmt='.', label=method_names[4])
    
    # rects1 = ax.bar(x_ticks - 2*width, means[:, 0, i], width, yerr=stds[:, 0, i], label=method_names[0])
    # rects2 = ax.bar(x_ticks - 1*width, means[:, 1, i], width, yerr=stds[:, 1, i], label=method_names[1])
    # rects3 = ax.bar(x_ticks , means[:, 2, i], width, yerr=stds[:, 2, i], label=method_names[2])
    # rects4 = ax.bar(x_ticks + 1*width, means[:, 3, i], width, yerr=stds[:, 3, i], label=method_names[3])
    # rects5 = ax.bar(x_ticks + 2*width, means[:, 4, i], width, yerr=stds[:, 4, i], label=method_names[4])
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([100, 200, 300, 400]) # , fontdict={'fontsize': 20}
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_yticklabels([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_ylim(0.3, 1.1)
    ax.set_title(metric_names[i], fontsize=15)
    if i in [2, 3]:
        ax.set_xlabel('Sample Size', fontsize=15)
    if i == 2:
        ax.legend(loc='best') 
        
# ax.legend(loc='best', bbox_to_anchor=(1.05,1.0)) 




# ================== six cases results ===================
def six_cases():
    pass

metric_names = ['Accuracy', 'Precision', 'Recall', r'$F_1$ score']
regression_names = ['ACP', 'Lasso_AIC', 'Lasso_BIC', 'Lasso_CV', 'Random Forest']
classification_names = ['ACP', r'Logistic Regression with $l_1$ norm', 'Random Forest']

urls = paths[[16, 15, 11, 13, 12, 17]]
thresholds = [0.9, 0.75, 0.9, 0.8, 0.9, 0.9]

data_names = ['Linear with independent features', 'Linear with correlated features', 'Linear with cross terms', 'Hierarchical function', 'Binary classification', 'Multi-classification']    

x_ticks = np.array([200, 400, 600, 800])
width = 25

fig, axes = plt.subplots(2, 3, sharey=True)  
idx = 0 
for i in range(2):
    for j in range(3):
        dats = np.load(urls[idx])   
        
        if idx < 4:
            result = regression_result(dats, thresholds[idx]) # (method_types, repetitions, metrics)        
            means = result.mean(axis=1)    
            stds = result.std(axis=1)
            
            rects1 = axes[i, j].bar(x_ticks - 2*width, means[0, :], width, yerr=stds[0, :], label=regression_names[0])
            rects2 = axes[i, j].bar(x_ticks - 1*width, means[1, :], width, yerr=stds[1, :], label=regression_names[1])
            rects3 = axes[i, j].bar(x_ticks , means[2, :], width, yerr=stds[2, :], label=regression_names[2])
            rects4 = axes[i, j].bar(x_ticks + 1*width, means[3, :], width, yerr=stds[3, :], label=regression_names[3])
            rects5 = axes[i, j].bar(x_ticks + 2*width, means[4, :], width, yerr=stds[4, :], label=regression_names[4])
            
            axes[i, j].set_xticks(x_ticks)
            axes[i, j].set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
            axes[i, j].set_ylim(0, 1)
            axes[i, j].set_title(data_names[idx], fontdict={'fontsize': 20})                        
        else:
            result = classification_result(dats, thresholds[idx]) 
            means = result.mean(axis=1)    
            stds = result.std(axis=1)
            
            rects1 = axes[i, j].bar(x_ticks - 1*width, means[0, :], width, yerr=stds[0, :], label=classification_names[0])
            rects2 = axes[i, j].bar(x_ticks, means[1, :], width, yerr=stds[1, :], label=classification_names[1])
            rects3 = axes[i, j].bar(x_ticks + 1*width , means[2, :], width, yerr=stds[2, :], label=classification_names[2])
            
            axes[i, j].set_xticks(x_ticks)
            axes[i, j].set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
            axes[i, j].set_ylim(0, 1)
            axes[i, j].set_title(data_names[idx], fontdict={'fontsize': 20})
        
        idx += 1
        
axes[0, 2].legend(loc='center right', bbox_to_anchor=(1.3,1.0))
axes[1, 2].legend(loc='lower right', bbox_to_anchor=(1.3,1.0))



# ======================= table ============================================   
def table():
    pass
# linear m=100, 200, 300, 400, n=50, [7, 16, 19, 21], threshold=0.9
# linear correlated m=100, 200, 300, 400, n=50, [6, 15, 18, 20], threshold=0.75

# linear m=100, n=300, [5], threshold=0.9
# linear m=200, n=300, [9], threshold=0.9

# linear correlated m=100,200, n=1000, [4, 8], threshold=[0.9, 0.7]

# cross term m=200, n=50, [14], threshold=0.9, reward=linear_func

dats = np.load(paths[11])

_, _, n = dats.shape
num_support = 8
y_true = np.zeros(n, dtype=int)
y_true[:num_support] = 1

theta = dats[:,0,:]
y_pred_rl1 = np.where(theta > 0.9, 1, 0)

y_pred_aic = np.zeros((50, n))
nums = np.sum(dats[:, 1, :] != 0, axis=1)
for i, num in enumerate(nums):
    idx = np.where(dats[i, 1, :] != 0)[0]
    y_pred_aic[i, idx] = 1
      
y_pred_bic = np.zeros((50, n))
nums = np.sum(dats[:, 2, :] != 0, axis=1)
for i, num in enumerate(nums):
    idx = np.where(dats[i, 2, :] != 0)[0]
    y_pred_bic[i, idx] = 1
    
y_pred_cv = np.zeros((50, n))
nums = np.sum(dats[:, 3, :] != 0, axis=1)
for i, num in enumerate(nums):
    idx = np.where(dats[i, 3, :] != 0)[0]
    y_pred_cv[i, idx] = 1  

rf_means = np.mean(dats[:, 4, :], axis=1)
y_pred_rf = np.zeros((50, n))
for i, mean in enumerate(rf_means):
    y_pred_rf[i, :] =np.where(dats[i, 4, :] > mean, 1, 0)


# y_pred_cv = np.zeros((50, n))
# nums = np.sum(dats[:, 1, :] != 0, axis=1)
# for i, num in enumerate(nums):
#     idx = np.where(dats[i, 1, :] != 0)[0]
#     y_pred_cv[i, idx] = 1  


# means = np.mean(dats[:, 2, :], axis=1)
# y_pred_rf = np.zeros((50, n))
# for i, mean in enumerate(means):
#     y_pred_rf[i, :] =np.where(dats[i, 2, :] > mean, 1, 0)


metric_names = ['Accuracy', 'Precision', 'Recall', r'$F_1$ score']
method_names = ['ACP', 'Lasso_AIC', 'Lasso_BIC', 'Lasso_CV', 'Random Forest']
# method_names = ['ACP', r'Logistic Regression with $l_2$ norm', 'Random Forest']


results = pd.DataFrame(np.zeros((4, 5)), index=metric_names)
# results = pd.DataFrame(np.zeros((5, 3)), index=metric_names)

for i, y_preds in enumerate([y_pred_rl1, y_pred_aic, y_pred_bic, y_pred_cv, y_pred_rf]):
# for i, y_preds in enumerate([y_pred_rl1, y_pred_cv, y_pred_rf]):
    result = np.array([compute_metrics(y_true, y_pred) for y_pred  in y_preds])
    
    results.loc['Accuracy', i] = f'{result[:, 0].mean():.3f}'+'$\pm$'+f'{result[:, 0].std():.3f}'+' & '
    # results.loc['Balanced_accuracy', i] = f'{result[:, 1].mean():.3f}'+'-'+f'{result[:, 1].std():.3f}'    
    results.loc['Precision', i] = f'{result[:, 1].mean():.3f}'+'$\pm$'+f'{result[:, 1].std():.3f}'+' & '
    results.loc['Recall', i] = f'{result[:, 2].mean():.3f}'+'$\pm$'+f'{result[:, 2].std():.3f}'+' & '
    results.loc[r'$F_1$ score', i] = f'{result[:, 3].mean():.3f}'+'$\pm$'+f'{result[:, 3].std():.3f}'+' & '
 
results.columns = method_names

# ========================================================================================================

methods = [y_pred_rl1, y_pred_aic, y_pred_bic, y_pred_cv, y_pred_rf]

def barplot(methods, method_names):
    results = np.zeros((50, len(methods), len(metric_names)))
    # for i, cms in enumerate([cm1, cm_aic, cm_bic, cm_cv, cm_rf]):
    #     results[:, i, :] = np.array([compute_metrics(cm) for cm in cms])
    for i, y_preds in enumerate(methods):
        results[:, i, :] = np.array([compute_metrics(y_true, y_pred) for y_pred  in y_preds])


    x_ticks = np.array([200, 400, 600, 800, 1000])
    width = 25



    fig = plt.figure() 
    means = results.mean(axis=0)    
    stds = results.std(axis=0)
    ax = fig.add_subplot()
    rects1 = ax.bar(x_ticks - 2*width, means[0, :], width, yerr=stds[0, :], label=method_names[0])
    rects2 = ax.bar(x_ticks - 1*width, means[1, :], width, yerr=stds[1, :], label=method_names[1])
    rects3 = ax.bar(x_ticks , means[2, :], width, yerr=stds[2, :], label=method_names[2])
    rects4 = ax.bar(x_ticks + 1*width, means[3, :], width, yerr=stds[3, :], label=method_names[3])
    rects5 = ax.bar(x_ticks + 2*width, means[4, :], width, yerr=stds[4, :], label=method_names[4])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    # ax.set_yticks([0, 0.5, 0.8, 0.9, 1])
    # ax.grid(False, axis='x')        
    # ax.set_title(data_names[i], fontdict={'fontsize': 20}) 


def errorbar(methods, method_names):
    results = np.zeros((50, len(methods), len(metric_names)))
    for i, y_preds in enumerate(methods):
        results[:, i, :] = np.array([compute_metrics(y_true, y_pred) for y_pred  in y_preds])


    x_ticks = np.array([200, 400, 600, 800, 1000])
    width = 20


    fig = plt.figure() 
    means = results.mean(axis=0)    
    stds = results.std(axis=0)
    ax = fig.add_subplot()
    rects1 = ax.errorbar(x_ticks - 2*width, means[0, :], yerr=stds[0, :], fmt='.', elinewidth=5, ms=9,  label=method_names[0])
    rects2 = ax.errorbar(x_ticks - 1*width, means[1, :], yerr=stds[1, :], fmt='.', elinewidth=5, ms=9,  label=method_names[1])
    rects3 = ax.errorbar(x_ticks , means[2, :], yerr=stds[2, :], fmt='.', elinewidth=5, ms=9,  label=method_names[2])
    rects4 = ax.errorbar(x_ticks + 1*width, means[3, :], yerr=stds[3, :], fmt='.', elinewidth=5, ms=9,  label=method_names[3])
    rects5 = ax.errorbar(x_ticks + 2*width, means[4, :], yerr=stds[4, :], fmt='.', elinewidth=5, ms=9,  label=method_names[4])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
    ax.set_ylim(0, 1)
    ax.legend(loc='best')





#========================== real data results ==========================
# spam [23]
# crime(new) [2(24)]
# madelon [22]
# cnae-9(scale)(new) [0(1)(25)]


urls = paths[[23, 24, 22, 25]]
nums = [16, 50, 20, 55]


dats = np.load('spam_cv_npg.npy')

means = dats.mean(axis=0)
stds = dats.std(axis=0)

num = 50
# print(plt.style.available)
# plt.style.use('default')
# plt.style.use('grayscale')
# plt.style.use('seaborn-paper')
# plt.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
# plt.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label=r'Logistic Regression with $l_2$ norm')
# plt.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Random Forest')
# plt.errorbar(range(1, num), means[1:num, 3], yerr=stds[1:num, 3], linestyle='--', fmt='.', label='All Features')
# plt.xticks(range(1, num, 2))
# plt.legend(loc='lower right')


plt.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
plt.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label='Lasso_AIC')
plt.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Lasso_BIC')
plt.errorbar(range(1, num), means[1:num, 3], yerr=stds[1:num, 3], linestyle='--', fmt='.', label='Lasso_CV')
plt.errorbar(range(1, num), means[1:num, 4], yerr=stds[1:num, 4], linestyle='--', fmt='.', label='Random Forest')
plt.errorbar(range(1, num), means[1:num, 5], yerr=stds[1:num, 5], linestyle='--', fmt='.', label='All Features')
plt.xticks(range(1, num, 2))
plt.legend(loc='lower right')


plt.plot(range(1, num), means[1:num, 0], label='ACP')
plt.plot(range(1, num), means[1:num, 1], label=r'Logistic Regression with $l_2$ norm')
plt.plot(range(1, num), means[1:num, 2], label='Random Forest')
plt.plot(range(1, num), means[1:num, 3], label='All Features')
plt.xticks(range(1, num, 5))
plt.legend(loc='best')


def real_data():
    pass

fig = plt.figure()

ax0 = fig.add_subplot(2, 2, 1)
dats = np.load('spam_cv_npg.npy')

means = dats.mean(axis=0)
stds = dats.std(axis=0)
num = 30
ax0.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
ax0.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label=r'Logistic Regression with $l_2$ norm')
ax0.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Random Forest')
# ax0.errorbar(range(1, num), means[1:num, 3], yerr=stds[1:num, 3], linestyle='--', fmt='.', label='All Features')
ax0.axhline(means[1:num, 3][0], linestyle='--', label='All Features', color='black')
ax0.set_xticks(range(1, num+1, 2))
ax0.legend(loc='best')
ax0.set_title('Spambase', fontsize=20)
ax0.set_ylabel('Accuracy', fontsize=20)



ax1 = fig.add_subplot(2, 2, 2)
dats = np.load('z-crime_cv_npg_new.npy')

means = dats.mean(axis=0)
stds = dats.std(axis=0)
num = 30
ax1.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
ax1.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label='Lasso_AIC')
ax1.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Lasso_BIC')
ax1.errorbar(range(1, num), means[1:num, 3], yerr=stds[1:num, 3], linestyle='--', fmt='.', label='Lasso_CV')
ax1.errorbar(range(1, num), means[1:num, 4], yerr=stds[1:num, 4], linestyle='--', fmt='.', label='Random Forest')
# ax1.errorbar(range(1, num), means[1:num, 5], yerr=stds[1:num, 5], linestyle='--', fmt='.', label='All Features')
ax1.axhline(means[1:num, 5][0], linestyle='--', label='All Features', color='black')
ax1.set_xticks(range(1, num+1, 2))
ax1.legend(loc='best')
ax1.set_title('Communities and Crime', fontsize=20)
ax1.set_ylabel(r'$R^2$', fontsize=20)



ax2 = fig.add_subplot(2, 2, 3)
dats = np.load('madelon_cv_npg.npy')

means = dats.mean(axis=0)
stds = dats.std(axis=0)
num = 30
ax2.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
ax2.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label=r'Logistic Regression with $l_2$ norm')
ax2.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Random Forest')
# ax2.errorbar(range(1, num), means[1:num, 3], yerr=stds[1:num, 3], linestyle='--', fmt='.', label='All Features')
ax2.axhline(means[1:num, 3][0], linestyle='--', label='All Features', color='black')
ax2.set_xticks(range(1, num+1, 2))
ax2.legend(loc='best')
ax2.set_title('Madelon', fontsize=20)
ax2.set_ylabel('Accuracy', fontsize=20)
ax2.set_xlabel('Numbers of selected covariates', fontsize=20)



ax3 = fig.add_subplot(2, 2, 4)
dats = np.load('zz-cnae-9_cv_npg_new.npy')

means = dats.mean(axis=0)
stds = dats.std(axis=0)
num = 55
ax3.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
ax3.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label=r'Logistic Regression with $l_2$ norm')
ax3.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Random Forest')
# ax3.errorbar(range(1, num), means[1:num, 3], yerr=stds[1:num, 3], linestyle='--', fmt='.', label='All Features')
ax3.axhline(means[1:num, 3][0], linestyle='--', label='All Features', color='black')
ax3.set_xticks(range(1, num+1, 5))
ax3.legend(loc='best')
ax3.set_title('CNAE-9', fontsize=20)
ax3.set_ylabel('Accuracy', fontsize=20)
ax3.set_xlabel('Numbers of selected covariates', fontsize=20)