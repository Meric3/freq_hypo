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
sys.path.append('../hypotension')
# -

# test 중 ..



1+1

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

# +
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

args.config = '../hypotension/config/config.yml'
opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))

# # load dataset

# +
import os
import pickle5 as pickle

import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class dnn_dataset(torch.utils.data.Dataset):
    def __init__(self, features, abp, ecg, ple, co2, target, caseid, invasive, multi):
        self.features = features
        self.invasive, self.multi = invasive, multi
        self.abp, self.ecg, self.ple, self.co2 = abp, ecg, ple, co2
        self.target = target
        self.caseid = caseid
        
    def __getitem__(self, index):
        if self.features == 'abp':
            return np.float32( np.array ( self.abp[index] ) ), np.float32(self.target[index]), np.float32(self.caseid[index]) 
        if self.features == 'ecg':
            return np.float32( np.array ( self.ecg[index] ) ), np.float32(self.target[index]), np.float32(self.caseid[index]) 
        if self.features == 'ple':
            return np.float32( np.array ( self.ple[index] ) ), np.float32(self.target[index]), np.float32(self.caseid[index]) 
        if self.features == 'co2':
            return np.float32( np.array ( self.co2[index] ) ), np.float32(self.target[index]), np.float32(self.caseid[index]) 
        if self.invasive == True:
            if self.multi == True: # Invasive multi-channel model
                return np.float32( np.vstack (( np.array ( self.abp[index] ),
                                                np.array ( self.ecg[index] ),
                                                np.array ( self.ple[index] ),
                                                np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index]), np.float32(self.caseid[index])
            else: # Invasive mono-channel model (arterial pressure-only model)
                return np.float32( np.array ( self.abp[index] ) ), np.float32(self.target[index]), np.float32(self.caseid[index]) 
        else:
            if self.multi == True: # Non-invasive multi-channel model
                return np.float32( np.vstack (( np.array ( self.ecg[index] ),
                                                np.array ( self.ple[index] ),
                                                np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index]), np.float32(self.caseid[index])
            else: # Non-invasive mono-channel model (photoplethysmography-only model)
                return np.float32( np.array ( self.ple[index] ) ), np.float32(self.target[index]), np.float32(self.caseid[index]) 

    def __len__(self):
        return len(self.target)


# -

random_key = 77

# +
train_ratio = 0.6 # Size for training dataset
valid_ratio = 0.1 # Size for validation dataset
test_ratio = 0.3 # Size for test dataset
invasive = opt['invasive']
multi = opt['multi']
pred_lag = opt['pred_lag']
task_target = opt['task_target']

processed_dir = opt['data_path']
batch_size=opt['batch_size']
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
raw_records = raw_records[(raw_records['map']>=20)&(raw_records['map']<=160)].reset_index(drop=True) # Exclude abnormal range

if opt['print_option_1'] == True: 
    print ( 'Invasive: {}\nMulti: {}\nPred lag: {}\n'.format ( invasive, multi, pred_lag ))
records = raw_records.loc[ ( raw_records['input_length']==30 ) &
                            ( raw_records['pred_lag']==pred_lag ) ]

records = records [ records.columns.tolist()[-1:] + records.columns.tolist()[:-1] ]

if opt['print_option_1'] == True: 
    print ( 'N of total records: {}'.format ( len ( records ) ))

split_records = {}
for phase in ['train', 'valid', 'test']:
    split_records[phase] = records[records['caseid'].isin(cases[phase])].reset_index(drop=True)

    if opt['print_option_1'] == True: 
        print ('- N of {} records: {}'.format ( phase, len ( split_records[phase] )))

if opt['print_option_1'] == True: 
    print ( '' )

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


# -

# # fourier

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dnn_inputs.shape

batch_size

dnn_inputs[0].shape

dnn_inputs_.shape

# +
current_loss = 0
output_stack = 0
loaders = loader['train']
for idx, (dnn_inputs, dnn_target, tp) in enumerate(loaders):
#     import pdb; pdb.set_trace()
    dnn_inputs, dnn_target= dnn_inputs.to(device), dnn_target.to(device)        
   
    for index_ in range(4):
        if index_ == 0:
#             import pdb; pdb.set_trace()
            dnn_inputs_ = make_fourier(dnn_inputs[:,index_,:]).to(device).unsqueeze(1)
        else:
            dnn_inputs_ = torch.cat([dnn_inputs_, make_fourier(dnn_inputs[:,index_,:]).to(device).unsqueeze(1)], dim=1)
    
    if idx == 0:
        output_stack = dnn_inputs_
    else:
        output_stack = torch.cat([output_stack, dnn_inputs_], dim=0)

# output_stack.extend ( np.array ( dnn_inputs_.cpu().T[0] ) )

output_stack =  np.array ( output_stack.cpu() )

weight_list = []
for index in range(output_stack.shape[1]):
    weight_list.append(output_stack[:,index,:][~np.isnan(output_stack[:,index,:])].sum())

whole_ = 0
for weight in weight_list:
    whole_ += weight

weight_list/whole_
# -

weight_list = [0.52, 0.007, 0.24, 0.22] 

dataset['test'].ecg[0]

dataset['test'].ecg[0] * 0.52

weight_list/whole_

output2 = output_stack[:,0,:][~np.isnan(output_stack[:,0,:])] 

output_stack[0,:,0]

output_stack[0,1::2,0]

output_stack[0,:,::2]

output_stack[0,0,1::2]

output_stack[0,0,::2]

output_stack[0,0,:4]

output2.sum()



np.sum(output_stack[:,0,:])







 np.vstack (( np.array ( self.abp[index] ),np.array ( self.ecg[index] ),np.array ( self.ple[index] ),np.array ( self.co2[index] ) ) ) 












