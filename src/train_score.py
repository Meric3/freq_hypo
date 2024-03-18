import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import random
from random import randint
import argparse
from timeit import default_timer as timer
from datetime import timedelta
import tqdm, yaml
from datasets.load_dataset import load_dataset 
from models.make_model import make_model
from utils.log import make_log
from trains.load_train import load_train


dfcases = pd.read_csv("https://api.vitaldb.net/cases")

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--exp', type = str, default='temp')
parser.add_argument('--config', type = str, default='./config/config.yml')
parser.add_argument('--project', type = str, default='temp')
parser.add_argument("--local_save", default=True, action="store_true")
parser.add_argument('--base_path', type = str, default='/hdd1/mjh319/saves/')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--tensorboard", default=False, action="store_true")
parser.add_argument("--load_model", default=False, action="store_true")
parser.add_argument('--load_model_ckpt', type = str, default='.')

parser.add_argument("--invasive", default=False, action="store_true")
parser.add_argument("--multi", default=False, action="store_true")
parser.add_argument('--load_dataset_method', type = str, default='numpy4')
parser.add_argument('--train_method', type = str, default='none')
parser.add_argument('--features', type = str, default='none')
parser.add_argument('--model', type = str, default='1d_cnn')
parser.add_argument('--m2', type=int, default=-1)
parser.add_argument('--pred_lag', type=int, default=300)

args = parser.parse_args()

opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))

if args.local_save == True:
    log, log_dir_path, summary = make_log(opt)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("data {} device {}".format(opt["data_path"], device))

train, evaluate = load_train(opt, log, device)

criterion = nn.BCELoss()
finish_list = []
for pred in opt['pred_lag_list']:
    for multi in opt['multi_list']:
        for invasive in opt['invasive_list']:
            opt['invasive'] = invasive
            opt['multi'] = multi
            opt['pred_lag'] = pred
            print(invasive, multi , pred)
            
            start = timer()           

            # Read dataset
            test_auc_by_best_valid_auc_list = []
            for tests in range(opt['total_exp']): 
                random_key = randint(0, 100000) 

                epoch_loss, epoch_auc = {}, {}

                for phase in [ 'train', 'valid', 'test' ]:
                    epoch_loss[phase], epoch_auc[phase] = [], []
       
            
                start_d = timer() 
                loader = load_dataset(opt, log, random_key, dfcases)
                end_d = timer()

                model = make_model(opt, device)


                optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])

                best_loss, best_auc = 99999.99999, 0.0
                if opt['print_option_1'] == True:
                    tqdm_ = tqdm.tqdm(range(opt['max_epoch']))
                else:
                    tqdm_ = range(opt['max_epoch'])
                
                for epoch in tqdm_:
                    if epoch == 0:
                        start_t = timer() 
                    elif epoch == 1:
                        end_t = timer()

                    target_stack, output_stack, case_stack, input_stack = {}, {}, {}, {}
                    current_loss, current_auc = {}, {}
                    for phase in [ 'train', 'valid', 'test' ]:
                        target_stack[phase], output_stack[phase] =  [], []
                        current_loss[phase], current_auc[phase] = 0.0, 0.0


                    current_loss['train'] = train(epoch, model,  loader['train'], criterion, optimizer, device, opt)  
                    epoch_loss['train'].append ( current_loss['train'] ) 

                    for phase in [ 'valid', 'test']:
                        current_loss[phase], target_stack[phase], output_stack[phase], case_stack[phase], input_stack[phase] = \
                        evaluate(model,  loader[phase], criterion, optimizer, device, opt)       


                    log_label = {}
                    for phase in ['valid', 'test']:
                        current_auc[phase] = roc_auc_score ( target_stack[phase], output_stack[phase] )
                        epoch_auc[phase].append ( current_auc[phase] )

                    best = ''
                    if abs(current_auc['valid']) > abs(best_auc):
                        best = '< ! >'
                        last_saved_epoch = epoch
                        best_auc = abs(current_auc['valid'])
                        test_auc_by_best_valid_auc = abs(current_auc['test'])
                        save_path = str(log_dir_path) + "/" +str(opt['pred_lag'])\
                                + str(opt['multi']) + str(opt['invasive']) + "_"+str(tests)+".pt"
                        torch.save({'model':model.state_dict(),
                                   'target_stack':target_stack,
                                   'output_stack':output_stack,
                                    'case_stack':case_stack,
                                   'input_stack':input_stack,
                                   'model':model}, save_path)
                    
                    if opt['print_option_1'] == True:
                        tqdm_.set_description("random_key[{}][e:{}][last:{}][exp:{}][auc/best:{:.4f}/{:.4f}][tr_l:{:.2f}][te_l:{:.2f}][va_l:{:.2f}]"
                                         .format(random_key, epoch, last_saved_epoch, tests,current_auc['test'],\
                                                 best_auc, current_loss['train'],current_loss['test'],current_loss['valid']))

                                            
                test_auc_by_best_valid_auc_list.append(test_auc_by_best_valid_auc)

                
                
            end = timer()
            finish_list.append('Finish Ex[{}],D[{}],T[{}],I:[{}]M:[{}][{}],[AUC:{:.4f}][mean:{:.4f}][std:{:.4f}]'.\
                                      format(timedelta(seconds=end-start),timedelta(seconds=end_d-start_d),\
                                             timedelta(seconds=end_t-start_t), invasive, multi, pred, \
                                             abs(current_auc['test']),\
                                             np.mean(test_auc_by_best_valid_auc_list),\
                                             np.std(test_auc_by_best_valid_auc_list)))

for text in finish_list:
    print(text)                                          

