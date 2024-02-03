import os
# import yaml
from pathlib import Path
from datetime import timezone, timedelta, datetime
# from tensorboardX import SummaryWriter
import logging
import pickle

import numpy as np
import torch

from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve, average_precision_score

        
def make_fourier(inputs):
    n = 3000
    Fs =100.0
    T = n/Fs
    k = np.arange(n)
    freq = k/T
    freq = freq[range(int(n/2))]
    
    signal = inputs.cpu().numpy()
    signal_list = []
    for i in range(inputs.shape[0]):
        y = signal[i,:]
        Y = np.fft.fft(y)/n 
        Y = Y[range(int(n/2))]
        signal_list.append(np.hstack([abs(Y),abs(Y)]))
    signal_np = np.asarray(signal_list)
    signal_np = torch.from_numpy(signal_np)
    signal_np= signal_np.type(torch.FloatTensor)
    return signal_np
    
    dnn_inputs = torch.cat([dnn_inputs,signal_np.to(device).unsqueeze(1)], dim=1)
    


def make_log(opt):
    tz = timezone(timedelta(hours=9))
    now = datetime.now(tz)
    mon = format(now.month, '02')
    day = format(now.day, '02')
    h = format(now.hour, '02')
    m = format(now.minute, '02')
    s = format(now.second, '02')
    date = mon+day
    now = h+m+s
    print("Time {}, {}".format(date, now))

    if opt['local_save'] == True:
        log_dir_path = Path(opt['log_base_path'], opt['project_name'])
        log_dir_path.mkdir(parents=True, exist_ok=True)

        log_dir_path = Path(log_dir_path, date)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        tf_path = Path(log_dir_path, "tf")
        tf_path.mkdir(parents=True, exist_ok=True)

        log_dir_path = Path(log_dir_path, opt['exp'] +now)
        log_dir_path.mkdir(parents=True, exist_ok=True)    

        tf_path = Path(tf_path, now)
        tf_path.mkdir(parents=True, exist_ok=True)

        print("path {} ".format(log_dir_path))
        log_path = str(log_dir_path) + "/log.txt"
        print("log_path {} ".format(log_path))

    cp_src = "cp -r ../deeplearning " + str(log_dir_path) + "/"
    os.system(cp_src)

    summary = SummaryWriter(tf_path)

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(formatter)
    log.addHandler(fileHandler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    log.info("-"*99)
    log.info(now)
    log.info("-"*99)
    for name in opt:
        log.info("{} : {}".format(name, opt[name]))    
    
    return log, log_dir_path


def result(target_stack,  output_stack):
    auc = roc_auc_score ( target_stack, output_stack )
    aupr = average_precision_score( target_stack, output_stack )

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
    
    return auc,aupr, ppv_equal, npv_equal, sensi_equal, speci_equal, ppv_90, npv_90, sensi_90, speci_90







