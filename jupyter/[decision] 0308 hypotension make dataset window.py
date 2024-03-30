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
# # %load_ext autoreload
# # %autoreload 2

import sys 
sys.path.append('/home/mjh319/workspace/hypotension')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

from pathlib import Path

import numpy as np
import random

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse
from random import randint

import yaml
# import time
from datetime import datetime 

import os


# implementation
# from dataset import Make_hypo_dataset
from datasets.make_load_dataset import make_load_dataset 
# from model import Net, Informer
from models.make_model import make_model
from utils.util import result

from utils.log import make_log
from trains.load_train import load_train

import pandas

import pickle

import os
import pickle5 as pickle

import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# +
parser = argparse.ArgumentParser()

parser.add_argument('--exp', type = str, default='temp')
parser.add_argument('--config', type = str, default='../config/config.yml')
parser.add_argument('--project', type = str, default='temp')
parser.add_argument("--local_save", default=True, action="store_false")
parser.add_argument('--base_path', type = str, default='/hdd1/mjh319/saves/')


parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--tensorboard", default=False, action="store_true")
parser.add_argument("--load_model", default=False, action="store_true")
parser.add_argument('--load_model_ckpt', type = str, default='.')

parser.add_argument("--invasive", default=False, action="store_true")
parser.add_argument("--multi", default=False, action="store_true")
parser.add_argument('--modification', type = str, default='none')
parser.add_argument('--features', type = str, default='none')
parser.add_argument('--m2', type=int, default=-1)
parser.add_argument('--pred_lag', type=int, default=300)

args = parser.parse_args([])
# -

args.multi = True
args.invasive = True
args.model = 'logistic'

args.config = '/home/mjh319/workspace/hypotension/config/config.yml'
opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))


def make_fourier(inputs, range_):
    n = range_
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



class dnn_dataset(torch.utils.data.Dataset):
    def __init__(self, features, abp, ecg, ple, co2, target, caseid, invasive, multi):
        self.features = features
        self.invasive, self.multi = invasive, multi
        self.abp, self.ecg, self.ple, self.co2 = abp, ecg, ple, co2
        self.target = target
        self.caseid = caseid
        
    def __getitem__(self, index):
        if self.invasive == True:
            if self.multi == True: # Invasive multi-channel model
                return np.float32( np.vstack (( np.array ( self.abp[index] ),
                                                np.array ( self.ecg[index] ),
                                                np.array ( self.ple[index] ),
                                                np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index]), np.float32(self.caseid[index])
            else: # Invasive mono-channel model (arterial pressure-only model)
                return np.float32( np.array ( self.abp[index] ) ), np.float32(self.target[index])      , np.float32(self.caseid[index]) 
        else:
            if self.multi == True: # Non-invasive multi-channel model
                return np.float32( np.vstack (( np.array ( self.ecg[index] ),
                                                np.array ( self.ple[index] ),
                                                np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index]), np.float32(self.caseid[index])
            else: # Non-invasive mono-channel model (photoplethysmography-only model)
                return np.float32( np.array ( self.ple[index] ) ), np.float32(self.target[index]), np.float32(self.caseid[index])

    def __len__(self):
        return len(self.target)


# +
base_path_base = '/hdd2/mjh319/data/hypotension/window_processed/'
# /hdd2/mjh319/data/hypotension/window_processed
processed_dir_base ='/hdd2/mjh319/data/hypotension/processed/'

data_path_ = Path(base_path_base)
data_path_.mkdir(parents=True, exist_ok=True)

pred_lag_list = [300, 600, 900]
input_length_ = [10,20,40,50,60 ]
random_key = 777
for inputs_ in input_length_:
    base_path = base_path_base + str(inputs_) + "/"
    data_path_ = Path(base_path)
    data_path_.mkdir(parents=True, exist_ok=True)
    processed_dir = processed_dir_base+ str(inputs_) + "/"
    for pred_lag in pred_lag_list:

        train_ratio = 0.99 # Size for training dataset
        valid_ratio = 0.05 # Size for validation dataset
        test_ratio = 0.05 # Size for test dataset
        invasive = opt['invasive']
        multi = opt['multi']
        task_target = 'hypo'

        batch_size = 1
        num_workers=opt['num_workers']

        file_list = np.char.split ( np.array ( os.listdir(processed_dir) ), '.' )
        case_list = []
        for caseid in file_list:
            case_list.append ( int ( caseid[0] ) )
        if opt['print_option_1'] == True: 
            print ( 'N of total cases: {}'.format ( len ( case_list ) ) )


        cases = {}
        cases['train'], cases['valid+test'] = train_test_split ( case_list,
                                                                test_size=(valid_ratio+test_ratio),
                                                                random_state=random_key )
        cases['valid'], cases['test'] = train_test_split ( cases['valid+test'],
                                                          test_size=(test_ratio/(valid_ratio+test_ratio)),
                                                          random_state=random_key )

        for phase in [ 'train', 'valid', 'test' ]:
            if opt['print_option_1'] == True: 
                print ( "- N of {} cases: {}".format(phase, len(cases[phase])) )

        for idx, caseid in enumerate(case_list):
            filename = processed_dir + str ( caseid ) + '.pkl'
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
                data['caseid'] = [ caseid ] * len ( data['abp'] )
                raw_records = raw_records.append ( pd.DataFrame ( data ) ) if idx > 0 else pd.DataFrame ( data )

        nan_list = set()
        for x in ['abp','ecg','ple','co2']:
            j = 0
            for i in raw_records[x]:
                if np.isnan(i).any() == True:
                    nan_list.add(j)
                j += 1

        nan_list = list(nan_list)
        indexes_to_keep = set(range(raw_records.shape[0])) - set(nan_list)
        raw_records = raw_records.take(list(indexes_to_keep))
        raw_records = raw_records[(raw_records['map']>=20)&(raw_records['map']<=160)].reset_index(drop=True) # Exclude abnormal range

    #     -----------------------------------------------------------

        case_list_2 = []
        new_records = pd.DataFrame()
        for idx, case in enumerate(case_list):
            if case not in case_list_2:
                t = raw_records[raw_records['caseid']==case]
                sample_lists = list(t[t['hypo']==1].index)
                if len(sample_lists)> 0:
                    if len(sample_lists) > 10:
                        selects = 4
                    else:
                        selects = 1

                    sample_lists = list(random.sample(sample_lists, selects))
                    t2 = t.loc[sample_lists]

                    if new_records.empty == False:
                        new_records = new_records.append ( pd.DataFrame ( t2 ) )
                    else :
                        new_records = pd.DataFrame ( t2 )

                normal = list(t[t['hypo']==0].index)
                if len(normal)> 0:
                    ratios = int(len(normal)/3)
                    normal = normal[ratios:2*ratios]

                    if len(normal) > 50:
                        selects =50
                    else:
                        selects = len(normal)


                    sample_lists = list(random.sample(normal, selects))
                    t3 = t.loc[sample_lists]
                    if new_records.empty == False:
                        new_records = new_records.append ( pd.DataFrame ( t3 ) )
                    else :
                        new_records = pd.DataFrame ( t3 )
                else:
                    continue
                case_list_2.append(case)

        cases['train'], cases['valid+test'] = train_test_split ( list(new_records.index),
                                                                test_size=(valid_ratio+test_ratio),
                                                                random_state=random_key )
        cases['valid'], cases['test'] = train_test_split ( cases['valid+test'],
                                                          test_size=(test_ratio/(valid_ratio+test_ratio)),
                                                          random_state=random_key )

        split_records = {}
        for phase in ['train', 'valid', 'test']:
            split_records[phase] = new_records.loc[cases[phase]].reset_index(drop=True)

    #     -----------------------------------------------------------    


    #     records = raw_records.loc[ ( raw_records['input_length']==30 ) &
    #                                 ( raw_records['pred_lag']==pred_lag ) ]

    #     records = records [ records.columns.tolist()[-1:] + records.columns.tolist()[:-1] ]


    #     split_records = {}
    #     for phase in ['train', 'valid', 'test']:
    #         split_records[phase] = records[records['caseid'].isin(cases[phase])].reset_index(drop=True)

    #     -----------------------------------------------------------    
        ext = {}
        for phase in [ 'train', 'valid', 'test' ]:
            ext[phase] = {}
            for x in [ 'abp', 'ecg', 'ple', 'co2', 'hypo', 'map', 'caseid' ]:
                ext[phase][x] = split_records[phase][x]

        dataset, loader = {}, {}
        epoch_loss, epoch_auc = {}, {}

        for phase in [ 'train', 'valid', 'test' ]:
            dataset[phase] = dnn_dataset ( opt['features'],
                                            ext[phase]['abp'],
                                            ext[phase]['ecg'],
                                            ext[phase]['ple'],
                                            ext[phase]['co2'],
                                            ext[phase][task_target],
                                          ext[phase]['caseid'],
                                            invasive = invasive, multi = multi )
            loader[phase] = torch.utils.data.DataLoader(dataset[phase],
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        shuffle = True if phase == 'train' else False )

        # fourier

        abp_time = []
        abp_freq = []
        ecg_time = []
        ecg_freq = []
        ple_time = []
        ple_freq = []
        co2_time = []
        co2_freq = []
        target = []
        caseid = []
        dataset_list = {}

        dataset_list['abp_time'] = abp_time
        dataset_list['abp_freq'] = abp_freq
        dataset_list['ecg_time'] = ecg_time
        dataset_list['ecg_freq'] = ecg_freq
        dataset_list['ple_time'] = ple_time
        dataset_list['ple_freq'] = ple_freq
        dataset_list['co2_time'] = co2_time
        dataset_list['co2_freq'] = co2_freq
        dataset_list['target'] = target
        dataset_list['caseid'] = caseid

        loaders = loader['train']
        with torch.no_grad():
            for idx_, (dnn_inputs, dnn_target, case_id) in enumerate(loaders):


                dnn_inputs, dnn_target = dnn_inputs, dnn_target

                if torch.isnan(dnn_inputs).any().item() is True:
                    print("error nan")
                    continue

        #         import pdb;pdb.set_trace()

                dataset_list['caseid'].extend(np.array(int(case_id)).reshape(-1,1))
                dataset_list['target'].extend(np.array(int(dnn_target)).reshape(-1,1))
                for idx, key in enumerate(dataset_list):
                    if idx < 8:
                        index_ = int(idx/2)
                        if int(idx/2) == idx/2:
                            dataset_list[key].extend(np.array ( dnn_inputs[:,index_,:]))
                        else:
                            dataset_list[key].extend(np.array ( make_fourier(dnn_inputs[:,index_,:], dnn_inputs[:,index_,:].shape[1])))
                    else:
                        break



        for idx, key in enumerate(dataset_list):
            dataset_list[key] = np.array(dataset_list[key])



        for idx, key in enumerate(dataset_list):
            print(key, dataset_list[key].shape)

        transform_dict = {}

        for key in dataset_list:
            if "_" in key:
                print(key)
                transform_dict[key] = [dataset_list[key].mean(), dataset_list[key].std()]

        dataset_list['transform_dict'] = transform_dict

        for idx, key in enumerate(dataset_list):
            print(key)
            if key != 'transform_dict':
                t = pandas.DataFrame(dataset_list[key])
                each_data_path = base_path+key+'_'+str(pred_lag)+".pkl"
                with open(each_data_path, 'wb') as file:    
                    pickle.dump(dataset_list[key], file)
            else:
                each_data_path = base_path+"transform_dict"+'_'+str(pred_lag)+".pkl"
                with open(each_data_path, 'wb') as file:    
                    pickle.dump(dataset_list['transform_dict'], file)

        abp_time = []
        abp_freq = []
        ecg_time = []
        ecg_freq = []
        ple_time = []
        ple_freq = []
        co2_time = []
        co2_freq = []
        patient_index = []
        caseid = []
        transform_dict = []

        dataset_list_ = {}

        dataset_list_['abp_time'] = abp_time
        dataset_list_['abp_freq'] = abp_freq
        dataset_list_['ecg_time'] = ecg_time
        dataset_list_['ecg_freq'] = ecg_freq
        dataset_list_['ple_time'] = ple_time
        dataset_list_['ple_freq'] = ple_freq
        dataset_list_['co2_time'] = co2_time
        dataset_list_['co2_freq'] = co2_freq
        dataset_list_['target'] = target
        dataset_list_['caseid'] = caseid
        dataset_list['transform_dict'] = transform_dict

        for idx, key in enumerate(dataset_list_): 
            if key != 'transform_dict':
                path = base_path+key+'_'+str(pred_lag)+".pkl"
                with open(path, 'rb') as file:    
                    dataset_list_[key] = pickle.load(file)
            else:
                path = base_path+"transform_dict"+'_'+str(pred_lag)+".pkl"
                with open(path, 'rb') as file:    
                    dataset_list_['transform_dict'] = pickle.load(file)
# -

dnn_inputs[:,index_,:].shape[1]

filename

t['abp'][0].shape








