import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import os
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score



paths = os.listdir('./npg_results')
paths = np.array(paths)

# 0  'm100_n1000_linear_correlated_var0.9_signal1_sigma0.5_step1500_batch64_linearfunc_0.9.npy',
# 1  'm100_n300_linear_signal1_sigma0.5_step200_batch64_linearfunc_0.9.npy',
# 2  'm100_n50_linear_correlated_var0.9_signal1_sigma0.5_step500_batch64_linearfunc_0.7.npy',
# 3  'm100_n50_linear_signal1_sigma0.5_step200_batch64_linearfunc_0.9.npy',
# 4  'm200_n1000_linear_correlated_var0.9_signal1_sigma0.5_step1500_batch64_ridgefunc_0.7.npy',
# 5  'm200_n1000_linear_signal1_sigma0.5_step1000_batch64_ridgefunc_0.9.npy',
# 6 'm200_n300_logistic_sigma0.5_step450_batch64_mlp_reward_0.75.npy',
# 7 'm200_n30_intersection_signal1_sigma1_step100_batch64_mlp_reward_4.8h.npy',
# 8 'm200_n50_binary_classification_signal1_sigma0.5_step150_batch64_mlp_reward_14.5h.npy',
# 9 'm200_n50_hierarchical_sigma0.5_step150_batch64_mlp_reward.npy',
# 10 'm200_n50_intersection_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 11 'm200_n50_linear_correlated_var0.9_signal1_sigma0.5_step750_batch64_linearfunc_0.75.npy',
# 12 'm200_n50_linear_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 13 'm200_n50_make_classification_step200_batch64_mlp_reward_19.5h.npy',
# 14 'm300_n50_linear_correlated_var0.9_signal1_sigma0.5_step1000_batch64_linearfunc_0.8.npy',
# 15 'm300_n50_linear_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 16 'm400_n50_linear_correlated_var0.9_signal1_sigma0.5_step500_batch64_linearfunc_0.75.npy',
# 17 'm400_n50_linear_signal1_sigma0.5_step100_batch64_linearfunc_0.9.npy',
# 18 'madelon_cv_npg.npy',
# 19 'spam_cv_npg.npy'
# 20 'z-crime_cv_npg_new.npy'
# 21 'zz-cnae-9_cv_npg_new.npy'

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
        result.append([compute_metrics(y_true, y_pred) for y_pred in y_preds])
    
    result = np.array(result)
    
    return result



#========================== synthetic results ==========================


## ============ linear cases =========================
urls = paths[[2, 11, 14, 16]]         # linear correlated =====Figure 3=====
thresholds = [0.75, 0.75, 0.75, 0.75]
# urls = paths[[3, 12, 15, 17]]       # linear independent ====Figure B1==== 
# thresholds = [0.9, 0.9, 0.9, 0.9] 
tmp = np.load(os.path.join('./npg_results', urls[0]))

_, _, n = tmp.shape
num_support = 8
y_true = np.zeros(n, dtype=int)
y_true[:num_support] = 1


results = []
for url, threshold in zip(urls, thresholds):
    dats = np.load(os.path.join('./npg_results', url))
    results.append(regression_result(dats, threshold))
    
results = np.array(results)   # (n_samples, methods_types, runs, metrics)
    
means = results.mean(axis=2)   # (n_samples, method_types, metrics) 
stds = results.std(axis=2)

    
metric_names = ['Accuracy', 'Precision', 'Recall', r'$F_1$ score']
method_names = ['ACP', 'Lasso_AIC', 'Lasso_BIC', 'Lasso_CV', 'Random Forest', 'TVS']


path_tvs = os.listdir('./TVS_code_data')
# 0:'binary.txt',
# 1:'hierarchical.txt',
# 2:'intersection.txt',
# 3:'linear_correlated.txt',
# 4:'m100_n300_linear.txt',
# 5:'m100_n50_linear.txt',
# 6:'m200_n300_linear.txt',
# 7:'m200_n50_linear.txt',
# 8:'m300_n50_linear.txt',
# 9:'m400_n50_linear.txt',
# 10:'multi_class.txt',
# 11:'TVS.R',
# 12:'tvs_crime_scale.npy',
# 13:'tvs_crime_scale.txt',
# 14:'tvs_madelon_scale.npy',
# 15:'tvs_madelon_scale.txt',
# 16:'TVS_realdata.R',
# 17:'tvs_spam.npy',
# 18:'tvs_spam.txt',
# 19:'tvs_spam_scale.npy',
# 20:'tvs_spam_scale.txt'

tvs_results_mean = []
tvs_results_std = []
urls = [5, 7, 8, 9]  # independent
for url in urls:
    dat = np.loadtxt(os.path.join('./TVS_code_data', path_tvs[url]))
    tvs_metrics = [[compute_metrics(y_true, y_pred)] for y_pred in dat]
    tvs_metrics = np.array(tvs_metrics)
    
    tvs_results_mean.append(tvs_metrics.mean(axis=0))
    tvs_results_std.append(tvs_metrics.std(axis=0))



add_tvs_path = os.listdir('./add')
# 0:'m100_n300_correlated.txt',
# 1:'m100_n50_correlated.txt',
# 2:'m200_n300_correlated.txt',
# 3:'m200_n300_intersect.txt',
# 4:'m200_n50_correlated.txt',
# 5:'m300_n50_correlated.txt',
# 6:'m400_n50_correlated.txt'

tvs_results_mean = []
tvs_results_std = []
urls = [1, 4, 5, 6]  # correlated
for url in urls:
    dat = np.loadtxt(os.path.join('./add', add_tvs_path[url]))
    tvs_metrics = [[compute_metrics(y_true, y_pred)] for y_pred in dat]
    tvs_metrics = np.array(tvs_metrics)
    
    tvs_results_mean.append(tvs_metrics.mean(axis=0))
    tvs_results_std.append(tvs_metrics.std(axis=0))




tvs_results_mean = np.array(tvs_results_mean)  # (n_samples, 1, metrics)
tvs_results_std = np.array(tvs_results_std)  # (n_samples, 1, metrics)

means = np.concatenate((means, tvs_results_mean), axis=1)
stds = np.concatenate((stds, tvs_results_std), axis=1)


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
    ax.errorbar(x_ticks + 2*width, means[:, 5, i], yerr=stds[:, 5, i], linestyle='--', fmt='.', label=method_names[5])
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([100, 200, 300, 400]) # , fontdict={'fontsize': 20}
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_yticklabels([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_ylim(0.3, 1.1)
    ax.set_title(metric_names[i], fontsize=15)
    if i in [2, 3]:
        ax.set_xlabel('Sample Size', fontsize=15)
    if i == 2:
        ax.legend(loc='best') 
        
# ax.legend(loc='best', bbox_to_anchor=(1.05,1.0)) 



# ================== six cases results, Figure 2 ===================


metric_names = ['Accuracy', 'Precision', 'Recall', r'$F_1$ score']
regression_names = ['ACP', 'Lasso_AIC', 'Lasso_BIC', 'Lasso_CV', 'Random Forest', 'TVS']
classification_names = ['ACP', r"Logistic Regression with $l_1$ norm", 'Random Forest', 'FSTD']

urls = paths[[12, 11, 7, 9, 8, 13]]
thresholds = [0.9, 0.75, 0.9, 0.8, 0.9, 0.9]

data_names = ['Linear with independent features', 'Linear with correlated features', 'Linear with cross terms', 'Hierarchical function', 'Binary classification', 'Multi-classification']    

x_ticks = np.array([200, 400, 600, 800])
width = 25


tvs_results_mean = []
tvs_results_std = []
tvs_urls = [7, 3, 2, 1, 0]
for url in tvs_urls:
    tmp = np.loadtxt(os.path.join('./TVS_code_data', path_tvs[url]))
    tvs_metrics = [[compute_metrics(y_true, y_pred)] for y_pred in tmp]
    tvs_metrics = np.array(tvs_metrics)
    
    tvs_results_mean.append(tvs_metrics.mean(axis=0))
    tvs_results_std.append(tvs_metrics.std(axis=0))


tvs_results_mean = np.array(tvs_results_mean)  # (data_type, 1, metrics)
tvs_results_std = np.array(tvs_results_std)  # (data_type, 1, metrics)


fstd_binary = np.load('./fstd_binary.npy')
fstd_results = [compute_metrics(y_true, y_pred) for y_pred in fstd_binary]
fstd_results = np.array(fstd_results)
fstd_mean = fstd_results.mean(axis=0)
fstd_std = fstd_results.std(axis=0)


fig, axes = plt.subplots(2, 3, sharey=True)  
idx = 0 
for i in range(2):
    for j in range(3):
        dats = np.load(os.path.join('./npg_results', urls[idx]))   
        
        if idx < 4:
            result = regression_result(dats, thresholds[idx]) # (method_types, repetitions, metrics)        
            means = result.mean(axis=1)    
            stds = result.std(axis=1)
                        
            
            rects1 = axes[i, j].bar(x_ticks - 2*width, means[0, :], width, yerr=stds[0, :], label=regression_names[0])
            rects2 = axes[i, j].bar(x_ticks - 1*width, means[1, :], width, yerr=stds[1, :], label=regression_names[1])
            rects3 = axes[i, j].bar(x_ticks , means[2, :], width, yerr=stds[2, :], label=regression_names[2])
            rects4 = axes[i, j].bar(x_ticks + 1*width, means[3, :], width, yerr=stds[3, :], label=regression_names[3])
            rects5 = axes[i, j].bar(x_ticks + 2*width, means[4, :], width, yerr=stds[4, :], label=regression_names[4])
            rects6 = axes[i, j].bar(x_ticks + 3*width, tvs_results_mean[idx, 0, :], width, yerr=tvs_results_std[idx, 0, :], label=regression_names[5])
            
            
            axes[i, j].set_xticks(x_ticks)
            axes[i, j].set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
            axes[i, j].set_ylim(0, 1)
            axes[i, j].set_title(data_names[idx], fontdict={'fontsize': 20})                        
        else:
            result = classification_result(dats, thresholds[idx]) 
            means = result.mean(axis=1)    
            stds = result.std(axis=1)
            
            if idx == 4:
                rects1 = axes[i, j].bar(x_ticks - 2*width, means[0, :], width, yerr=stds[0, :], label=classification_names[0])
                rects2 = axes[i, j].bar(x_ticks - 1*width, means[1, :], width, yerr=stds[1, :], label=classification_names[1])
                rects3 = axes[i, j].bar(x_ticks, means[2, :], width, yerr=stds[2, :], label=classification_names[2])
                # rects4 = axes[i, j].bar(x_ticks + 1*width , tvs_results_mean[idx, 0, :], width, yerr=tvs_results_std[idx, 0, :], color='brown', label=regression_names[5])
                rects5 = axes[i, j].bar(x_ticks + 1*width , fstd_mean, width, yerr=fstd_std, label=classification_names[3])
            else:
                rects1 = axes[i, j].bar(x_ticks - 1*width, means[0, :], width, yerr=stds[0, :], label=classification_names[0])
                rects2 = axes[i, j].bar(x_ticks, means[1, :], width, yerr=stds[1, :], label=classification_names[1])
                rects3 = axes[i, j].bar(x_ticks + 1*width , means[2, :], width, yerr=stds[2, :], label=classification_names[2])
                
            
            axes[i, j].set_xticks(x_ticks)
            axes[i, j].set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
            axes[i, j].set_ylim(0, 1)
            axes[i, j].set_title(data_names[idx], fontdict={'fontsize': 20})
        
        idx += 1
        
axes[0, 2].legend(loc='center right', bbox_to_anchor=(1.4, 1.0))
axes[1, 1].legend(loc='lower right', bbox_to_anchor=(2.5, 1.0))




#========================== real data results, Figure 4 ==========================
# spam [23]
# crime(new) [2(24)]
# madelon [22]
# cnae-9(scale)(new) [0(1)(25)]

## TVS results
# 12:'tvs_crime_scale.npy',
# 13:'tvs_crime_scale.txt',
# 14:'tvs_madelon_scale.npy',
# 15:'tvs_madelon_scale.txt',
# 16:'TVS_realdata.R',
# 17:'tvs_spam.npy',
# 18:'tvs_spam.txt',
# 19:'tvs_spam_scale.npy',
# 20:'tvs_spam_scale.txt'



urls = paths[[19, 20, 18, 21]]



fig = plt.figure()

ax0 = fig.add_subplot(2, 2, 1)
dats = np.load(urls[0])  # 'spam_cv_npg.npy'

means = dats.mean(axis=0)
stds = dats.std(axis=0)


spam_tvs = np.loadtxt(os.path.join('./TVS_code_data', path_tvs[17]))   # 17, 19


num = 30
ax0.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
ax0.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label=r'Logistic Regression with $l_2$ norm')
ax0.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Random Forest')
ax0.errorbar(range(1, num), spam_tvs.mean(axis=0)[1:num], yerr=spam_tvs.std(axis=0)[1:num], linestyle='--', fmt='.', label='TVS')
ax0.axhline(means[1:num, 3][0], linestyle='--', label='All Features', color='black')
ax0.set_xticks(range(1, num+1, 2))
ax0.legend(loc='best')
ax0.set_title('Spambase', fontsize=20)
ax0.set_ylabel('Accuracy', fontsize=20)



ax1 = fig.add_subplot(2, 2, 2)
dats = np.load(urls[1])  # 'z-crime_cv_npg_new.npy'

means = dats.mean(axis=0)
stds = dats.std(axis=0)


crime_tvs = np.loadtxt(os.path.join('./TVS_code_data', path_tvs[12]))


num = 30
ax1.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
ax1.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label='Lasso_AIC')
ax1.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Lasso_BIC')
ax1.errorbar(range(1, num), means[1:num, 3], yerr=stds[1:num, 3], linestyle='--', fmt='.', label='Lasso_CV', color='purple')
ax1.errorbar(range(1, num), means[1:num, 4], yerr=stds[1:num, 4], linestyle='--', fmt='.', label='Random Forest', color='green')
ax1.errorbar(range(1, num), crime_tvs.mean(axis=0)[1:num], yerr=crime_tvs.std(axis=0)[1:num], linestyle='--', fmt='.', label='TVS', color='red')
ax1.axhline(means[1:num, 5][0], linestyle='--', label='All Features', color='black')
ax1.set_xticks(range(1, num+1, 2))
ax1.legend(loc='best')
ax1.set_title('Communities and Crime', fontsize=20)
ax1.set_ylabel(r'$R^2$', fontsize=20)



ax2 = fig.add_subplot(2, 2, 3)
dats = np.load(urls[2])   # 'madelon_cv_npg.npy'

means = dats.mean(axis=0)
stds = dats.std(axis=0)


madelon_tvs = np.loadtxt(os.path.join('./TVS_code_data', path_tvs[14]))


num = 30
ax2.errorbar(range(1, num), means[1:num, 0], yerr=stds[1:num, 0], linestyle='--', fmt='.', label='ACP')
ax2.errorbar(range(1, num), means[1:num, 1], yerr=stds[1:num, 1], linestyle='--', fmt='.', label=r'Logistic Regression with $l_2$ norm')
ax2.errorbar(range(1, num), means[1:num, 2], yerr=stds[1:num, 2], linestyle='--', fmt='.', label='Random Forest')
ax2.errorbar(range(1, num), madelon_tvs.mean(axis=0)[1:num], yerr=madelon_tvs.std(axis=0)[1:num], linestyle='--', fmt='.', label='TVS')
ax2.axhline(means[1:num, 3][0], linestyle='--', label='All Features', color='black')
ax2.set_xticks(range(1, num+1, 2))
ax2.legend(loc='best')
ax2.set_title('Madelon', fontsize=20)
ax2.set_ylabel('Accuracy', fontsize=20)
ax2.set_xlabel('Numbers of selected covariates', fontsize=20)



ax3 = fig.add_subplot(2, 2, 4)
dats = np.load(urls[3])  # 'zz-cnae-9_cv_npg_new.npy'

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







