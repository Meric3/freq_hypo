# Read dataset
import pdb

import os
import pickle5 as pickle

import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import random 

import torchvision.transforms as transforms


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
target = []
transform_dict = []
ehr = []

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
dataset_list_['transform_dict'] = transform_dict
dataset_list_['ehr']  = ehr
    
    
normal_i_m_feature_keys = ['abp_time', 'ple_time', 'ecg_time', 'co2_time']
normal_i_f_feature_keys = ['abp_time']
normal_f_m_feature_keys = ['ple_time','ecg_time','co2_time']
normal_f_f_feature_keys = ['ple_time']

freq_i_m_feature_keys = ['abp_freq', 'ple_freq', 'ecg_freq', 'co2_freq']
freq_i_f_feature_keys = ['abp_freq']
freq_f_m_feature_keys = ['ple_freq','ecg_freq','co2_freq']
freq_f_f_feature_keys = ['ple_freq']

all_i_m_feature_keys = ['abp_time', 'ple_time', 'ecg_time', 'co2_time', 'abp_freq', 'ple_freq', 'ecg_freq', 'co2_freq']
all_i_f_feature_keys = ['abp_time', 'abp_freq']
all_f_m_feature_keys = ['ple_time','ecg_time','co2_time', 'ple_freq','ecg_freq','co2_freq']
all_f_f_feature_keys = ['ple_time', 'ple_freq']


class dnn_dataset(torch.utils.data.Dataset):
    def __init__(self, opt, ext, invasive, multi):
#         self.dfcases = pd.read_csv("https://api.vitaldb.net/cases")
#         import pdb; pdb.set_trace()
        self.opt = opt
        self.invasive, self.multi = invasive, multi
        self.dataset_list_ = {}
        for idx, key in enumerate(dataset_list_): 
#             if key == 'ehr':   
#                 continue
            if key == 'transform_dict':                
                transform_dict = {}
                for trans_keys in ext[key]:
#                     0914 modified
                    if trans_keys == "ehr":
                        transform_dict[trans_keys] = ext[key][trans_keys]
                    else:
                        transform = []
                        transform.extend([
                                transforms.ToTensor(),
                                transforms.Normalize((ext[key][trans_keys][0]), (ext[key][trans_keys][1]))
                            ])
                        transform = transforms.Compose(transform)
                        transform_dict[trans_keys] = transform
                self.transform = transform_dict               
            elif key != 'target':
                self.dataset_list_[key] = ext[key]
            else:
                self.dataset_list_[key] = ext[key].reshape(-1)

       
        if self.invasive == True and self.multi == True and opt['features'] == 'none':
            self.feature_keys = normal_i_m_feature_keys
        elif self.invasive == True and self.multi == False and opt['features'] == 'none':
            self.feature_keys = normal_i_f_feature_keys
        elif self.invasive == False and self.multi == True and opt['features'] == 'none':
            self.feature_keys = normal_f_m_feature_keys  
        elif self.invasive == False and self.multi == False and opt['features'] == 'none':
            self.feature_keys = normal_f_f_feature_keys
        elif self.invasive == True and self.multi == True and opt['features'] == 'freq':
            self.feature_keys = freq_i_m_feature_keys
        elif self.invasive == True and self.multi == False and opt['features'] == 'freq':
            self.feature_keys = freq_i_f_feature_keys
        elif self.invasive == False and self.multi == True and opt['features'] == 'freq':
            self.feature_keys = freq_f_m_feature_keys  
        elif self.invasive == False and self.multi == False and opt['features'] == 'freq':
            self.feature_keys = freq_f_f_feature_keys
        elif self.invasive == True and self.multi == True and opt['features'] == 'all':
            self.feature_keys = all_i_m_feature_keys
        elif self.invasive == True and self.multi == False and opt['features'] == 'all':
            self.feature_keys = all_i_f_feature_keys
#             print("AAAA", self.feature_keys)
        elif self.invasive == False and self.multi == True and opt['features'] == 'all':
            self.feature_keys = all_f_m_feature_keys  
        elif self.invasive == False and self.multi == False and opt['features'] == 'all':
            self.feature_keys = all_f_f_feature_keys
        elif opt['features'] == 'ecg_time':
            self.feature_keys = ['ecg_time']
        elif opt['features'] == 'co2_time':
            self.feature_keys = ['co2_time']
        elif opt['features'] == 'ecg_freq':
            self.feature_keys = [ 'ecg_time', 'ecg_freq']
        elif opt['features'] == 'co2_freq':
            self.feature_keys =['co2_time', 'co2_freq']
        elif opt['features'] == 'ple_freq':
            self.feature_keys =['ple_freq']
        elif opt['features'] == 'abp_freq':
            self.feature_keys =['abp_freq']
#             print("ccc")
        else:
            raise
            
#         print("asdasd", self.feature_keys)
        
    def __getitem__(self, index):
#         import pdb; pdb.set_trace()
#         print("asdasd", self.feature_keys)
        for idx, feature in enumerate(self.feature_keys):
            if idx == 0:
                data_ = self.transform[feature](np.expand_dims(self.dataset_list_[feature][index,:], axis=0))
            else:
                data_ = torch.cat([data_,
                                     self.transform[feature](np.expand_dims(self.dataset_list_[feature][index,:], axis=0))], dim=0) 
                
                
#             0914 modied
        ehr = np.expand_dims(np.expand_dims(self.dataset_list_['ehr'][index,:], axis=0), axis=1)
        ehr = torch.permute(self.transform["ehr"](ehr), (1,2,0))
#     0914 modified
        return np.array(data_.squeeze().to(torch.float32)),\
                np.array(ehr.squeeze().to(torch.float32)),\
                np.array (np.float32(self.dataset_list_['target'][index])),\
                self.dataset_list_['caseid'][index]


    def __len__(self):
        return len(self.dataset_list_['target'])
    
    
def Make_hypo_dataset(opt, random_key, dfcases):
#     dfcases = pd.read_csv("https://api.vitaldb.net/cases")
    base_path = opt['data_path']
    train_ratio = 0.6 # Size for training dataset
    valid_ratio = 0.1 # Size for validation dataset
    test_ratio = 0.3 # Size for test dataset
    invasive = opt['invasive']
    multi = opt['multi']
    pred_lag = opt['pred_lag']
    
    processed_dir = opt['data_path']
    batch_size=opt['batch_size']
    num_workers=opt['num_workers']
    
    
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
    dataset_list_['transform_dict'] = transform_dict

    for idx, key in enumerate(dataset_list_): 
        if key != 'transform_dict':
            path = base_path+key+'_'+str(pred_lag)+".pkl"
            with open(path, 'rb') as file:    
                dataset_list_[key] = pickle.load(file)
        else:
            path = base_path+"transform_dict"+'_'+str(pred_lag)+".pkl"
            with open(path, 'rb') as file:    
                dataset_list_['transform_dict'] = pickle.load(file)
                
    data_lists = np.arange(dataset_list_['abp_time'].shape[0])

    
    
#     20220914 add EHR dataset

    ehr_stack = []
    col_ = opt['colum_ehr']
    for idx_, case in enumerate(dataset_list_['caseid']):
        t = dfcases[ dfcases['caseid']==case[0]]
        t = t[col_]
        t = np.asarray(t).astype(np.float32)
        ehr_stack.extend(t)
    dataset_list_['ehr'] = np.asarray(ehr_stack).astype(np.float32)

    transform = []
    transform.extend([
        transforms.ToTensor(),
        transforms.Normalize((dataset_list_['ehr'].mean(0)), (dataset_list_['ehr'].std(0)))
    ])
    transform = transforms.Compose(transform)
    dataset_list_['transform_dict']["ehr"] = transform
    
#     dataset_list_['ehr'] = None
#     dataset_list_['transform_dict']["ehr"] = None

        
#     20220914 add EHR dataset fin

    cases_lists = {}
    cases_lists['train'], cases_lists['valid+test'] = train_test_split ( data_lists,
                                                            test_size=(valid_ratio+test_ratio),
                                                            random_state=random_key )
    cases_lists['valid'], cases_lists['test'] = train_test_split ( cases_lists['valid+test'],
                                                      test_size=(test_ratio/(valid_ratio+test_ratio)),
                                                      random_state=random_key )
    

    ext = {}
    for phase in [ 'train', 'valid', 'test' ]:
        ext[phase] = {}
#         for x in [ 'abp_time', 'ecg_time', 'ple_time', 'co2_time', 'target', 'caseid',
#                  'abp_freq', 'ecg_freq', 'ple_freq', 'co2_freq']:
        for x in [ 'abp_time', 'ecg_time', 'ple_time', 'co2_time', 'target', 'caseid',
                 'abp_freq', 'ecg_freq', 'ple_freq', 'co2_freq', 'ehr']:
            ext[phase][x] = dataset_list_[x][cases_lists[phase]]
        ext[phase]['transform_dict'] = dataset_list_['transform_dict']
    
    
    dataset, loader = {}, {}
    epoch_loss, epoch_auc = {}, {}

    for phase in [ 'train', 'valid', 'test' ]:
        dataset[phase] = dnn_dataset( opt, ext[phase],
                                        invasive = invasive, multi = multi )
        loader[phase] = torch.utils.data.DataLoader(dataset[phase],
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    drop_last = True, 
                                                    shuffle = True if phase == 'train' else False )
        
    
    
    return loader    




