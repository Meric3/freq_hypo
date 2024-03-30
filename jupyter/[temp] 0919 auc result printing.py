# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: seg
#     language: python
#     name: seg
# ---

# +
# %load_ext autoreload
# %autoreload 2

import sys 
sys.path.append('/home/mjh319/workspace/3_hypotension_detection/3_hypo/')
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
import random
from random import randint
import argparse

from datasets.load_dataset import load_dataset 
from models.make_model import make_model
from utils.log import make_log
import yaml

from pathlib import Path

from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve

import matplotlib
import copy
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


# +
label_dict = {}
label_dict['300_False_False:comb_all'] = ''
label_dict['300_False_False:1d_cnn_none'] = ''
label_dict['300_True_True:comb_all'] = '5 min Combine Frequency Domain'
label_dict['300_True_True:1d_cnn_none'] = '5 min Only Time Domain'
label_dict['300_False_False:1d_cnn_ecg_time'] = ''
label_dict['300_False_False:comb_ecg_freq'] = ''
label_dict['300_False_False:1d_cnn_co2_time'] = ''
label_dict['300_False_False:comb_co2_freq'] = ''



label_dict['600_False_False:comb_all'] = ''
label_dict['600_False_False:1d_cnn_none'] = ''
label_dict['600_True_True:comb_all'] = ''
label_dict['600_True_True:1d_cnn_none'] = ''
label_dict['600_False_False:1d_cnn_ecg_time'] = ''
label_dict['600_False_False:comb_ecg_freq'] = ''
label_dict['600_False_False:1d_cnn_co2_time'] = ''
label_dict['600_False_False:comb_co2_freq'] = ''

label_dict['900_False_False:comb_all'] = ''
label_dict['900_False_False:1d_cnn_none'] = ''
label_dict['900_True_True:comb_all'] = ''
label_dict['900_True_True:1d_cnn_none'] = ''
label_dict['900_True_False:comb_all'] = 'Multi-domain'
label_dict['900_True_False:1d_cnn_none'] = 'Time-domain'
label_dict['900_False_False:1d_cnn_ecg_time'] = ''
label_dict['900_False_False:comb_ecg_freq'] = ''
label_dict['900_False_False:1d_cnn_co2_time'] = ''
label_dict['900_False_False:comb_co2_freq'] = ''

label_dict['300_False_True:comb_all'] = '5 min Combine Frequency Domain'
label_dict['300_False_True:1d_cnn_none'] = '5 min Only Time Domain'
label_dict['600_False_True:comb_all'] = '10 min Combine Frequency Domain'
label_dict['600_False_True:1d_cnn_none'] = '10 min Only Time Domain'
label_dict['900_False_True:comb_all'] = '15 min Combine Frequency Domain'
label_dict['900_False_True:1d_cnn_none'] = '15 min Only Time Domain'
# -

colot_dict_for_time = {}
colot_dict_for_time["300"] = 'purple'
colot_dict_for_time["600"] = 'teal'
colot_dict_for_time["900"] = 'crimson'

import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd


# + code_folding=[]
def result(target_stack,  output_stack):
    auc = roc_auc_score ( target_stack, output_stack )

    precision, recall, thmbps = precision_recall_curve(target_stack, output_stack)
    recall = np.nan_to_num(recall)
    precision = np.nan_to_num(precision)

    idx_90 = np.argmin(abs(recall-0.9))
    idx_equal = np.argmin(abs(recall-precision))        

    output_90 = output_stack > thmbps[idx_90]
    output_90 = output_90.astype(int)
    output_eq = output_stack > thmbps[idx_equal]
    output_eq = output_eq.astype(int)


    TN, FP, FN, TP =confusion_matrix(target_stack, output_eq).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    ppv_equal = PPV
    npv_equal = NPV
    sensi_equal = TPR
    speci_equal = TNR
    

    TN, FP, FN, TP =confusion_matrix(target_stack, output_90).ravel()
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ppv_90 = PPV
    npv_90 = NPV
    sensi_90 = TPR
    speci_90 = TNR
    
    return auc, ppv_equal, npv_equal

# +
parser = argparse.ArgumentParser()

parser.add_argument('--exp', type = str, default='temp')
parser.add_argument('--config', type = str, default='/home/mjh319/workspace/nnew_hypo/config/config.yml')
parser.add_argument('--project', type = str, default='temp')
parser.add_argument("--local_save", default=True, action="store_false")
parser.add_argument('--base_path', type = str, default='/hdd1/mjh319/saves/')


parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--tensorboard", default=False, action="store_true")
parser.add_argument("--load_model", default=False, action="store_true")
parser.add_argument('--load_model_ckpt', type = str, default='.')

parser.add_argument("--invasive", default=False, action="store_true")
parser.add_argument("--multi", default=False, action="store_true")
parser.add_argument('--load_dataset_method', type = str, default='normal')
parser.add_argument('--modification', type = str, default='none')
parser.add_argument('--features', type = str, default='none')
parser.add_argument('--model', type = str, default='1d_cnn')
parser.add_argument('--m2', type=int, default=-1)
parser.add_argument('--pred_lag', type=int, default=300)



args = parser.parse_args([])

opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))
    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

matplotlib.rcParams.update({'font.size': 30})
args.load_dataset_method = 'numpy4'

# -

test_list_for_dict_all = {}

# +
test_list_for_dict_all = {}
test_list = [0]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300, 600, 900]
multi_list = [True, False]
invasive_list = [True, False]
# pred_lag_list = [300]
# multi_list = [True]
# invasive_list = [True]

# test_name = ['time-series only', 'combine', 'frequency only', 'sd']
test_name = ['time-series only', 'combine']
# test_list_for_dict = {}

for pred in pred_lag_list:
    for multi in multi_list:
        for invasive in invasive_list:
            
            for test in test_list:
                if 0 == test:
                    base_path = '/hdd1/mjh319/saves/hypotension_0916_ehr_test/0916/comb_multi_100240/'
                    args.model = 'comb'
                    args.features = 'all'
                    opt.update(vars(args))
#                 elif 1 == test:
#                     base_path = '/hdd1/mjh319/saves/hypotension_0916_ehr_test/0916/time_multi_100242/'
#                     args.model = '1d_cnn'
#                     args.features = 'none'
#                     opt.update(vars(args))                                  
                else:
                    raise 

                    
                test_dict = {}
                

                opt['invasive'] = invasive
                opt['multi'] = multi
                opt['pred_lag'] = pred
                works = str(opt['pred_lag'])+"_"+ str(opt['invasive'])+"_"+str(opt['multi'])+":"
                test_dict['works'] = works
                test_dict['name'] = args.model + '_' + args.features
#                 random_key = 18235

                base = base_path + str(opt['pred_lag'])+str(opt['multi'])+str(opt['invasive']) +"_"
                target_ = []
                output_ = []
                for i in range(5):
                    model_path = base + str(i)+".pt"
                    load_ = torch.load(model_path)
                    target_.append(load_['target_stack']['test'])
                    output_.append(load_['output_stack']['test'])
                test_dict['target_stack'] = target_
                test_dict['output_stack'] = output_
                test_list_for_dict_all[works+test_dict['name']] = test_dict
                print(test_dict['works'],test_dict['name'])



# +
domains = []
features = []
times = []
aucs = []
ppvs = []
npvs = []
dict_ = test_list_for_dict_all.copy()
for name in dict_:
    target = dict_[name]['target_stack']
    output = dict_[name]['output_stack']
    auc_ = []
    npv_ = []
    ppv_ = []
    for idx in range(5):
        auc, ppv, npv = result( target[idx], output[idx])
        ppv_.append(ppv)
        npv_.append(npv)
        auc_.append(auc)
    
    auc_std = 2*np.std(auc_, axis=0)
    ppv_std = 2*np.std(ppv_, axis=0)
    npv_std = 2*np.std(npv_, axis=0)
    
    
#     auc          = np.mean(auc_)
    print(f'{name}\t{round(np.mean(auc_), 3)} + ({round(np.mean(auc_)-auc_std, 3)}~{round(np.mean(auc_)+auc_std, 3)})\
          \t{round(np.mean(ppv_), 3)} + ({round(np.mean(ppv_)-ppv_std, 3)}~{round(np.mean(ppv_)+ppv_std, 3)})\
          \t{round(np.mean(npv_), 3)} + ({round(np.mean(npv_)-npv_std, 3)}~{round(np.mean(npv_)+npv_std, 3)})')
    
    times.append(name.split(':')[0].split("_")[0])
    domains.append(label_dict[name.split(':')[1]])
    features.append(label_dict[name.split(':')[0][4:]])
    aucs.append(f'{round(np.mean(auc_), 3)} ({round(np.mean(auc_)-auc_std, 3)}-{round(np.mean(auc_)+auc_std, 3)})')
    ppvs.append(f'{round(np.mean(ppv_), 3)} ({round(np.mean(ppv_)-ppv_std, 3)}-{round(np.mean(ppv_)+ppv_std, 3)})')
    npvs.append(f'{round(np.mean(npv_), 3)} ({round(np.mean(npv_)-npv_std, 3)}-{round(np.mean(npv_)+npv_std, 3)})')
    
df = pd.DataFrame({'time':times, 'domain':domains, 'feature':features, 'auc':aucs, 'ppv':ppvs, 'npv':npvs})
df.to_csv('./hypotension/all.csv', index=False)

# +
# whole_dict = test_list_for_dict.copy()
# whole_dict.update(test_list_for_dict_one)
# test_list = [0,1,2,3]
# color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
# pred_lag_list = [300,600,900]
# multi_list = [False]
# invasive_list = [False]

# pred = 300

plt.figure(figsize=(15,15))
dict_ = test_list_for_dict_all.copy()
for idx in dict_:
    if idx.split(":")[0].split("_")[1] == "False" and idx.split(":")[0].split("_")[2] == "True":
        target = dict_[idx]['target_stack']
        output = dict_[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)

#         if idx.split("_")[0] == "300":
#         print(idx.split("_")[0])
        if idx.split(":")[1] == "comb_all":
            plt.plot(fpr_mean, tpr_mean, label = label_dict[idx], lw=3, color = colot_dict_for_time[idx.split("_")[0]])
#             plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = colot_dict_for_time[idx.split("_")[0]])
        else :
            plt.plot(fpr_mean, tpr_mean, label = label_dict[idx],linestyle='-.', lw=3, color = colot_dict_for_time[idx.split("_")[0]] )
#             plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = colot_dict_for_time[idx.split("_")[0]])
        
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Invaisive vs Non-invaisive 10 min prediction")
plt.legend()
plt.tight_layout()
# plt.savefig("./figure/hypotension/[AUC]invaisive_comparison.png", format = 'png')
plt.show()

# -

test_list_for_dict_one.keys()

dict_ = test_list_for_dict_noehr.copy()
dict_.update(test_list_for_dict_all)
for idx in dict_:
    print(idx.split(":")[0], idx.split(":")[1]) 600_True_False

# +
pred = "900"

plt.figure(figsize=(15,15))
# dict_ = test_list_for_dict_noehr.copy()
# dict_.update(test_list_for_dict_all)
dict_ = test_list_for_dict_all.copy()
dict_.update(test_list_for_dict_noehr)
for idx in dict_:
    if idx.split(":")[0] == "300_True_False" or idx.split(":")[0] == "300_True_True":
                
        target = dict_[idx]['target_stack']
        output = dict_[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)

        print(idx.split(":"), idx.split("_")[0])
        if idx.split(":")[0] == "300_True_True":
            color = "crimson"    
        else:
            color = "teal"    
            
        if idx.split(":")[0] == "300_True_True":
            label = "ABP with Multichannel & "
        else:
            label = "ABP & "
            
        if idx.split(":")[1] == "comb_all":
            label += "Multi-domain "
        else:
            label += "Time-domain"            
            
        if idx.split(":")[1] == "comb_all":
            plt.plot(fpr_mean, tpr_mean, label = label, lw=3, color = color)
        else :
            plt.plot(fpr_mean, tpr_mean, label = label,linestyle='-.', lw=3, color = color )

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.title("Invaisive vs Non-invaisive 10 min prediction")
plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]invaisive.png", format = 'png')
plt.show()

# -

dict_ = test_list_for_dict_all.copy()
for idx in dict_:
    print(idx)











# # co2 ecg

# +
test_list_for_dict_one = {}
test_list = [0,1,2,3]
# test_list = [0,1]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300,600,900]
# pred_lag_list = [300]
multi_list = [False]
invasive_list = [False]

for pred in pred_lag_list:
    for multi in multi_list:
        for invasive in invasive_list:
            
            for test in test_list:
                if 0 == test:
                    base_path = '/hdd1/mjh319/saves/hypotension_0916_ehr_test/0919/time_co2_134344/'
                    args.model = '1d_cnn'
                    args.features = 'co2_time'
                    opt.update(vars(args))
                elif 1 == test:
                    base_path = '/hdd1/mjh319/saves/hypotension_0916_ehr_test/0919/freq_co2_184627/'
                    args.model = 'comb'
                    args.features = 'co2_freq'
                    opt.update(vars(args))       
                elif 2 == test:
                    base_path = '//hdd1/mjh319/saves/hypotension_0916_ehr_test/0919/time_ecg_124658/'
                    args.model = '1d_cnn'
                    args.features = 'ecg_time'
                    opt.update(vars(args))
                elif 3 == test:
                    base_path = '/hdd1/mjh319/saves/hypotension_0916_ehr_test/0919/freq_ecg_124655/'
                    args.model = 'comb'
                    args.features = 'ecg_freq'
                    opt.update(vars(args))   
                else:
                    raise 

                    
                test_dict = {}
                

                opt['invasive'] = invasive
                opt['multi'] = multi
                opt['pred_lag'] = pred
                works = str(opt['pred_lag'])+"_"+ str(opt['invasive'])+"_"+str(opt['multi'])+":"
                test_dict['works'] = works
                test_dict['name'] = args.model + '_' + args.features
#                 random_key = 18235

                base = base_path + str(opt['pred_lag'])+str(opt['multi'])+str(opt['invasive']) +"_"
                target_ = []
                output_ = []
                for i in range(5):
                    model_path = base + str(i)+".pt"
                    load_ = torch.load(model_path)
                    target_.append(load_['target_stack']['test'])
                    output_.append(load_['output_stack']['test'])
                test_dict['target_stack'] = target_
                test_dict['output_stack'] = output_
                test_list_for_dict_one[works+test_dict['name']] = test_dict
                
                auc, _,_=result( target_[0], output_[0])
                print(test_dict['works'],test_dict['name'],auc)
#                 print()



# +
domains = []
features = []
times = []
aucs = []
ppvs = []
npvs = []
dict_ = test_list_for_dict_one.copy()
for name in dict_:
    target = dict_[name]['target_stack']
    output = dict_[name]['output_stack']
    auc_ = []
    npv_ = []
    ppv_ = []
    for idx in range(5):
        auc, ppv, npv = result( target[idx], output[idx])
        ppv_.append(ppv)
        npv_.append(npv)
        auc_.append(auc)
    
    auc_std = 2*np.std(auc_, axis=0)
    ppv_std = 2*np.std(ppv_, axis=0)
    npv_std = 2*np.std(npv_, axis=0)
    
    
#     auc          = np.mean(auc_)
    print(f'{name}\t{round(np.mean(auc_), 3)} + ({round(np.mean(auc_)-auc_std, 3)}~{round(np.mean(auc_)+auc_std, 3)})\
          \t{round(np.mean(ppv_), 3)} + ({round(np.mean(ppv_)-ppv_std, 3)}~{round(np.mean(ppv_)+ppv_std, 3)})\
          \t{round(np.mean(npv_), 3)} + ({round(np.mean(npv_)-npv_std, 3)}~{round(np.mean(npv_)+npv_std, 3)})')
    
    times.append(name.split(':')[0].split("_")[0])
    domains.append(label_dict[name.split(':')[1].split('_')[-1]])
    if 'ecg' in name.split(':')[1][5:] :
        features.append("ECG")
    elif 'co2' in name.split(':')[1][5:] :
        features.append("CO2")
    else:
        raise
    aucs.append(f'{round(np.mean(auc_), 3)} ({round(np.mean(auc_)-auc_std, 3)}-{round(np.mean(auc_)+auc_std, 3)})')
    ppvs.append(f'{round(np.mean(ppv_), 3)} ({round(np.mean(ppv_)-ppv_std, 3)}-{round(np.mean(ppv_)+ppv_std, 3)})')
    npvs.append(f'{round(np.mean(npv_), 3)} ({round(np.mean(npv_)-npv_std, 3)}-{round(np.mean(npv_)+npv_std, 3)})')
    
df = pd.DataFrame({'time':times, 'domain':domains, 'feature':features, 'auc':aucs, 'ppv':ppvs, 'npv':npvs})
df.to_csv('./hypotension/one.csv', index=False)
# -

dict_ = test_list_for_dict_one.copy()
for idx in dict_:
    if "co2" in idx.split(":")[1] and idx.split(":")[0].split("_")[0] == "300": 
        print( idx)

# +
# co2

pred = "900"
plt.figure(figsize=(15,15))
dict_ = test_list_for_dict_one.copy()
for idx in dict_:
    if "co2" in idx.split(":")[1] and idx.split(":")[0].split("_")[0] == pred: 
        target = dict_[idx]['target_stack']
        output = dict_[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)

#         if idx.split("_")[0] == "300":colot_dict_for_time["600"] = 'teal'
# colot_dict_for_time["900"] = 'crimson'
        if idx.split(":")[1] == "comb_co2_freq":
            plt.plot(fpr_mean, tpr_mean,  lw=3, color = "teal",label = "Multi-domain")
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "teal")
        else :
            plt.plot(fpr_mean, tpr_mean,linestyle='-.', lw=3, color = "crimson",label = "Time-domain" )
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "crimson")
        
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("CO2")
plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]co2.png", format = 'png')
plt.show()


# +
# co2

pred = "900"
plt.figure(figsize=(15,15))
dict_ = test_list_for_dict_one.copy()
for idx in dict_:
    if "ecg" in idx.split(":")[1] and idx.split(":")[0].split("_")[0] == pred: 
        print(idx)
        target = dict_[idx]['target_stack']
        output = dict_[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)

#         if idx.split("_")[0] == "300":colot_dict_for_time["600"] = 'teal'
# colot_dict_for_time["900"] = 'crimson'
        if idx.split(":")[1] == "comb_ecg_freq":
            plt.plot(fpr_mean, tpr_mean,  lw=3, color = "teal",label = "Multi-domain")
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "teal")
        else :
            plt.plot(fpr_mean, tpr_mean,linestyle='-.', lw=3, color = "crimson",label = "Time-domain" )
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "crimson")
        
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("ECG")
# plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]ecg.png", format = 'png')
plt.show()

# -



# # No ehrs

# +
test_list_for_dict_noehr = {}
test_list = [0,1]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300, 600, 900]
multi_list = [True, False]
invasive_list = [True, False]
# pred_lag_list = [300]
# multi_list = [True]
# invasive_list = [True]

# test_name = ['time-series only', 'combine', 'frequency only', 'sd']
test_name = ['time-series only', 'combine']
test_list_for_dict = {}

for pred in pred_lag_list:
    for multi in multi_list:
        for invasive in invasive_list:
            
            for test in test_list:
                if 0 == test:
                    base_path = '/hdd1/mjh319/saves/hypotension_0805/0805/comb_multi_121508/'
                    args.model = 'comb'
                    args.features = 'all'
                    opt.update(vars(args))
                elif 1 == test:
                    base_path = '/hdd1/mjh319/saves/hypotension_0805/0805/time_multi_121455/'
                    args.model = '1d_cnn'
                    args.features = 'none'
                    opt.update(vars(args))                                  
                else:
                    raise 

                    
                test_dict = {}
                

                opt['invasive'] = invasive
                opt['multi'] = multi
                opt['pred_lag'] = pred
                works = str(opt['pred_lag'])+"_"+ str(opt['invasive'])+"_"+str(opt['multi'])+":"
                test_dict['works'] = works
                test_dict['name'] = args.model + '_' + args.features
#                 random_key = 18235

                base = base_path + str(opt['pred_lag'])+str(opt['multi'])+str(opt['invasive']) +"_"
                target_ = []
                output_ = []
                for i in range(5):
                    model_path = base + str(i)+".pt"
                    load_ = torch.load(model_path)
                    target_.append(load_['target_stack']['test'])
                    output_.append(load_['output_stack']['test'])
                test_dict['target_stack'] = target_
                test_dict['output_stack'] = output_
                test_list_for_dict_noehr[works+test_dict['name']] = test_dict
                print(test_dict['works'],test_dict['name'])



# +
domains = []
features = []
times = []
aucs = []
ppvs = []
npvs = []
dict_ = test_list_for_dict_noehr.copy()
for name in dict_:
    target = dict_[name]['target_stack']
    output = dict_[name]['output_stack']
    auc_ = []
    npv_ = []
    ppv_ = []
    for idx in range(5):
        auc, ppv, npv = result( target[idx], output[idx])
        ppv_.append(ppv)
        npv_.append(npv)
        auc_.append(auc)
    
    auc_std = 2*np.std(auc_, axis=0)
    ppv_std = 2*np.std(ppv_, axis=0)
    npv_std = 2*np.std(npv_, axis=0)
    
    
#     auc          = np.mean(auc_)
    print(f'{name}\t{round(np.mean(auc_), 3)} + ({round(np.mean(auc_)-auc_std, 3)}~{round(np.mean(auc_)+auc_std, 3)})\
          \t{round(np.mean(ppv_), 3)} + ({round(np.mean(ppv_)-ppv_std, 3)}~{round(np.mean(ppv_)+ppv_std, 3)})\
          \t{round(np.mean(npv_), 3)} + ({round(np.mean(npv_)-npv_std, 3)}~{round(np.mean(npv_)+npv_std, 3)})')
    
    times.append(name.split(':')[0].split("_")[0])
    domains.append(label_dict[name.split(':')[1]])
    features.append(label_dict[name.split(':')[0][4:]])
    aucs.append(f'{round(np.mean(auc_), 3)} ({round(np.mean(auc_)-auc_std, 3)}-{round(np.mean(auc_)+auc_std, 3)})')
    ppvs.append(f'{round(np.mean(ppv_), 3)} ({round(np.mean(ppv_)-ppv_std, 3)}-{round(np.mean(ppv_)+ppv_std, 3)})')
    npvs.append(f'{round(np.mean(npv_), 3)} ({round(np.mean(npv_)-npv_std, 3)}-{round(np.mean(npv_)+npv_std, 3)})')
    
df = pd.DataFrame({'time':times, 'domain':domains, 'feature':features, 'auc':aucs, 'ppv':ppvs, 'npv':npvs})
df.to_csv('./hypotension/noehr.csv', index=False)

# +
# non invaisvie multi 

pred = "900"

plt.figure(figsize=(15,15))
dict_ = test_list_for_dict_noehr.copy()
for idx in dict_:
    if idx.split(":")[0].split("_")[1] == "False" and idx.split(":")[0].split("_")[2] == "True" and idx.split(":")[0].split("_")[0] ==pred:
        target = dict_[idx]['target_stack']
        output = dict_[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)

#         if idx.split("_")[0] == "300":
        if idx.split(":")[1] == "comb_all":
            plt.plot(fpr_mean, tpr_mean,  lw=3, color = "teal")
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "teal")
        else :
            plt.plot(fpr_mean, tpr_mean, linestyle='-.', lw=3, color = "crimson" )
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "crimson")
        
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("PLE + Multi-channel")
# plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]ple_multi.png", format = 'png')
plt.show()

# -

test_list_for_dict_all.keys()

# +
# non invaisvie  

pred = "900"

plt.figure(figsize=(15,15))
# whole_dict = test_list_for_dict_all.copy()
# whole_dict.update(test_list_for_dict_noehr)

dict_ = []
dict_.append(test_list_for_dict_all['900_False_False:comb_all'])
dict_.append(test_list_for_dict_noehr['900_False_False:1d_cnn_none'])
for i_, idx in enumerate(dict_):
#     if idx.split(":")[0].split("_")[1] == "False" and idx.split(":")[0].split("_")[2] == "False" and idx.split(":")[0].split("_")[0] ==pred:
#         print(idx)
#         dict_ = test_list_for_dict_all['900_False_False:comb_all']
    target = idx['target_stack']
    output = idx['output_stack']  
    auc_ = []
    fpr_ = []
    tpr_ = []
    for i in range(5):
        fpr, tpr, thresholds = roc_curve( target[i], output[i])
        fpr_.append(fpr)
        tpr_.append(tpr)
        auc_.append(roc_auc_score ( target[i], output[i]))

    fpr_mean    = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(5):
        fpr           = fpr_[i]
        tpr           = tpr_[i]
        interp_tpr    = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    tpr_mean     = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std      = 2*np.std(interp_tprs, axis=0)
    tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower    = tpr_mean-tpr_std
    auc          = np.mean(auc_)

#         if idx.split("_")[0] == "300":
    if i_ == 0:
        plt.plot(fpr_mean, tpr_mean,  lw=3, color = "teal")
        plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "teal")
    else :
        plt.plot(fpr_mean, tpr_mean, linestyle='-.', lw=3, color = "crimson" )
        plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = "crimson")
        
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("PLE")
# plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]ple.png", format = 'png')
plt.show()

# -

idx.keys()





















name.split(':')[1]

name.split(':')[0][4:]

name.split(':')

# +
label_dict = {}
label_dict['1d_cnn_none'] = 'Time-domain'
label_dict['comb_all'] = 'Multi-domain'
label_dict['True_True'] = 'ABP+Multichannel'
label_dict['True_False'] = 'ABP'
label_dict['False_True'] = 'PLE+Multichannel'
label_dict['False_False'] = 'PLE'
label_dict['time'] = 'Time-domain'
label_dict['freq'] = 'Multi-domain'

label_dict['300_False_False:comb_all'] = ''
label_dict['300_False_False:1d_cnn_none'] = ''
label_dict['300_True_True:comb_all'] = '5 min Combine Frequency Domain'
label_dict['300_True_True:1d_cnn_none'] = '5 min Only Time Domain'
label_dict['300_False_False:1d_cnn_ecg_time'] = ''
label_dict['300_False_False:comb_ecg_freq'] = ''
label_dict['300_False_False:1d_cnn_co2_time'] = ''
label_dict['300_False_False:comb_co2_freq'] = ''



label_dict['600_False_False:comb_all'] = ''
label_dict['600_False_False:1d_cnn_none'] = ''
label_dict['600_True_True:comb_all'] = ''
label_dict['600_True_True:1d_cnn_none'] = ''
label_dict['600_False_False:1d_cnn_ecg_time'] = ''
label_dict['600_False_False:comb_ecg_freq'] = ''
label_dict['600_False_False:1d_cnn_co2_time'] = ''
label_dict['600_False_False:comb_co2_freq'] = ''

label_dict['900_False_False:comb_all'] = ''
label_dict['900_False_False:1d_cnn_none'] = ''
label_dict['900_True_True:comb_all'] = ''
label_dict['900_True_True:1d_cnn_none'] = ''
label_dict['900_False_False:1d_cnn_ecg_time'] = ''
label_dict['900_False_False:comb_ecg_freq'] = ''
label_dict['900_False_False:1d_cnn_co2_time'] = ''
label_dict['900_False_False:comb_co2_freq'] = ''

label_dict['300_False_True:comb_all'] = '5 min Combine Frequency Domain'
label_dict['300_False_True:1d_cnn_none'] = '5 min Only Time Domain'
label_dict['600_False_True:comb_all'] = '10 min Combine Frequency Domain'
label_dict['600_False_True:1d_cnn_none'] = '10 min Only Time Domain'
label_dict['900_False_True:comb_all'] = '15 min Combine Frequency Domain'
label_dict['900_False_True:1d_cnn_none'] = '15 min Only Time Domain'
# -

# # auc figures



## metrics = ['auc', 'fpr', 'tpr', 'thresholds']
test_for_auc = test_list_for_dict_all
for i in test_for_auc:
    print(i)
    target = test_for_auc[i]['target_stack']
    output = test_for_auc[i]['output_stack']
    auc_ = []
    fpr_ = []
    tpr_ = []
    for i in range(5):
        fpr, tpr, thresholds = roc_curve( target[i], output[i])
        fpr_.append(fpr)
        tpr_.append(tpr)
        auc_.append(roc_auc_score ( target[i], output[i]))
        
    fpr_mean    = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(5):
        fpr           = fpr_[i]
        tpr           = tpr_[i]
        interp_tpr    = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    tpr_mean     = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std      = 2*np.std(interp_tprs, axis=0)
    tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower    = tpr_mean-tpr_std
    auc          = np.mean(auc_)
    
    plt.figure(figsize=(15,15))
    plt.plot(fpr_mean, tpr_mean)
    plt.fill_between(fpr_mean, tpr_lower, tpr_upper, color='grey', alpha=0.33,
                 label=r'$\pm$ 1 std. dev.')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(works)
    plt.legend()
    plt.show()


whole_dict = test_list_for_dict.copy()
whole_dict.update(test_list_for_dict_one)

color_list_for_time = ['darkorange','darkgreen','brown','purple','teal','crimson']

colot_dict_for_time = {}
colot_dict_for_time["300"] = 'purple'
colot_dict_for_time["600"] = 'teal'
colot_dict_for_time["900"] = 'crimson'

for i in whole_dict:
    if i.split(":")[0].split("_")[1] == "False" and i.split(":")[0].split("_")[2] == "True":
        print(i)

test_dict = {}
length_ = "300"
inputs = "600_False_True"
for i in whole_dict:
    print(i)
    if idx.split(":")[1].split("_")[0] == "comb":
        
    if idx.split("_")[0] == "600_False_True":
        
    if idx.split(":")[1].split("_")[0] == "comb":
        plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=3)
    else:
        plt.plot(fpr_mean, tpr_mean, label = str(idx),linestyle='-.', lw=3 )

# +
whole_dict = test_list_for_dict_all.copy()
# whole_dict.update(test_list_for_dict_one)
test_list = [0,1,2,3]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300,600,900]
multi_list = [False]
invasive_list = [False]

pred = 300

plt.figure(figsize=(15,15))

for idx in whole_dict:
    if idx.split(":")[0].split("_")[1] == "False" and idx.split(":")[0].split("_")[2] == "True":
        target = whole_dict[idx]['target_stack']
        output = whole_dict[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)

#         if idx.split("_")[0] == "300":
        if idx.split(":")[1] == "comb_all":
            plt.plot(fpr_mean, tpr_mean, label = label_dict[idx], lw=3, color = colot_dict_for_time[idx.split("_")[0]])
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = colot_dict_for_time[idx.split("_")[0]])
        else :
            plt.plot(fpr_mean, tpr_mean, label = label_dict[idx],linestyle='-.', lw=3, color = colot_dict_for_time[idx.split("_")[0]] )
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = colot_dict_for_time[idx.split("_")[0]])
        
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Invaisive vs Non-invaisive 10 min prediction")
plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]invaisive_comparison.png", format = 'png')
plt.show()


# +
whole_dict = test_list_for_dict.copy()
whole_dict.update(test_list_for_dict_one)
test_list = [0,1,2,3]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300,600,900]
multi_list = [False]
invasive_list = [False]

test_list_1 = ['False_False']


pred = 300

plt.figure(figsize=(15,15))

for idx in whole_dict:
    if test_list_1[0] in idx.split(":")[0] and idx.split("_")[0] == "600":
#     if idx.split("_")[0] == "300" and idx.split(":")[1] != "1d_cnn_none" and idx.split(":")[1] != "comb_all":
        target = whole_dict[idx]['target_stack']
        output = whole_dict[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)


        if "1d_cnn" in idx.split(":")[1] :
            plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2,linestyle='-.',)
        else:
            plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2)
#         plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = colot_dict_for_time[idx.split("_")[0]])

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
title = str(pred)
plt.title("Non-invaisive 10 min prediction")
# plt.legend()
plt.show()


# +
whole_dict = test_list_for_dict.copy()
whole_dict.update(test_list_for_dict_one)
test_list = [0,1,2,3]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300,600,900]
multi_list = [False]
invasive_list = [False]

test_list_1 = ['False_True']


pred = "600"

plt.figure(figsize=(15,15))

for idx in whole_dict:
    if test_list_1[0] in idx.split(":")[0] and idx.split("_")[0] == pred:
#     if idx.split("_")[0] == "300" and idx.split(":")[1] != "1d_cnn_none" and idx.split(":")[1] != "comb_all":
        target = whole_dict[idx]['target_stack']
        output = whole_dict[idx]['output_stack']  
        auc_ = []
        fpr_ = []
        tpr_ = []
        for i in range(5):
            fpr, tpr, thresholds = roc_curve( target[i], output[i])
            fpr_.append(fpr)
            tpr_.append(tpr)
            auc_.append(roc_auc_score ( target[i], output[i]))

        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(5):
            fpr           = fpr_[i]
            tpr           = tpr_[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(auc_)


        if "1d_cnn" in idx.split(":")[1] :
            plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2,linestyle='-.', color = 'crimson')
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'crimson')
        else:
            plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2, color = 'teal')
            plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'teal')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Multichannel with PLE 10 min prediction")
# plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]Multichannel_with_PLE.png", format = 'png')
plt.show()


# +
whole_dict = test_list_for_dict.copy()
whole_dict.update(test_list_for_dict_one)
test_list = [0,1,2,3]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300,600,900]
multi_list = [False]
invasive_list = [False]

test_list_1 = ['False_False']


pred = "600"

plt.figure(figsize=(15,15))

for idx in whole_dict:
    if test_list_1[0] in idx.split(":")[0] and idx.split("_")[0] == pred:
        if idx.split(":")[1] == "comb_all" or  idx.split(":")[1] == "1d_cnn_none" :

            target = whole_dict[idx]['target_stack']
            output = whole_dict[idx]['output_stack']  
            auc_ = []
            fpr_ = []
            tpr_ = []
            for i in range(5):
                fpr, tpr, thresholds = roc_curve( target[i], output[i])
                fpr_.append(fpr)
                tpr_.append(tpr)
                auc_.append(roc_auc_score ( target[i], output[i]))

            fpr_mean    = np.linspace(0, 1, 100)
            interp_tprs = []
            for i in range(5):
                fpr           = fpr_[i]
                tpr           = tpr_[i]
                interp_tpr    = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean     = np.mean(interp_tprs, axis=0)
            tpr_mean[-1] = 1.0
            tpr_std      = 2*np.std(interp_tprs, axis=0)
            tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
            tpr_lower    = tpr_mean-tpr_std
            auc          = np.mean(auc_)


            if "1d_cnn" in idx.split(":")[1] :
                plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2,linestyle='-.', color = 'crimson')
                plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'crimson')
            else:
                plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2, color = 'teal')
                plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'teal')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("PLE 10 min prediction")
# plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]PLE.png", format = 'png')
plt.show()


# +
whole_dict = test_list_for_dict.copy()
whole_dict.update(test_list_for_dict_one)
test_list = [0,1,2,3]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300,600,900]
multi_list = [False]
invasive_list = [False]

test_list_1 = ['ecg', 'co2']


pred = "600"

plt.figure(figsize=(15,15))

for idx in whole_dict:
    if idx.split("_")[0] == pred:
        if test_list_1[0] in idx.split(":")[1]:
            target = whole_dict[idx]['target_stack']
            output = whole_dict[idx]['output_stack']  
            auc_ = []
            fpr_ = []
            tpr_ = []
            for i in range(5):
                fpr, tpr, thresholds = roc_curve( target[i], output[i])
                fpr_.append(fpr)
                tpr_.append(tpr)
                auc_.append(roc_auc_score ( target[i], output[i]))

            fpr_mean    = np.linspace(0, 1, 100)
            interp_tprs = []
            for i in range(5):
                fpr           = fpr_[i]
                tpr           = tpr_[i]
                interp_tpr    = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean     = np.mean(interp_tprs, axis=0)
            tpr_mean[-1] = 1.0
            tpr_std      = 2*np.std(interp_tprs, axis=0)
            tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
            tpr_lower    = tpr_mean-tpr_std
            auc          = np.mean(auc_)


            if "1d_cnn" in idx.split(":")[1] :
                plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2,linestyle='-.', color = 'crimson')
                plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'crimson')
            else:
                plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2, color = 'teal')
                plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'teal')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("ECG 10 min prediction")
# plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]ECG.png", format = 'png')
plt.show()


# +
whole_dict = test_list_for_dict.copy()
whole_dict.update(test_list_for_dict_one)
test_list = [0,1,2,3]
color_list = ['darkorange','darkgreen','brown','purple','teal','crimson']
pred_lag_list = [300,600,900]
multi_list = [False]
invasive_list = [False]

test_list_1 = ['ecg', 'co2']


pred = "600"

plt.figure(figsize=(15,15))

for idx in whole_dict:
    if idx.split("_")[0] == pred:
        if test_list_1[1] in idx.split(":")[1]:
            target = whole_dict[idx]['target_stack']
            output = whole_dict[idx]['output_stack']  
            auc_ = []
            fpr_ = []
            tpr_ = []
            for i in range(5):
                fpr, tpr, thresholds = roc_curve( target[i], output[i])
                fpr_.append(fpr)
                tpr_.append(tpr)
                auc_.append(roc_auc_score ( target[i], output[i]))

            fpr_mean    = np.linspace(0, 1, 100)
            interp_tprs = []
            for i in range(5):
                fpr           = fpr_[i]
                tpr           = tpr_[i]
                interp_tpr    = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean     = np.mean(interp_tprs, axis=0)
            tpr_mean[-1] = 1.0
            tpr_std      = 2*np.std(interp_tprs, axis=0)
            tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
            tpr_lower    = tpr_mean-tpr_std
            auc          = np.mean(auc_)

            if "1d_cnn" in idx.split(":")[1] :
                plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2,linestyle='-.', color = 'crimson')
                plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'crimson')
            else:
                plt.plot(fpr_mean, tpr_mean, label = str(idx), lw=2, color = 'teal')
                plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=0.23, color = 'teal')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("CO2 10 min prediction")
plt.legend()
plt.tight_layout()
plt.savefig("./figure/hypotension/[AUC]CO2.png", format = 'png')
plt.show()

# -








