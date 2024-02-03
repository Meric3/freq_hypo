import torch
import torch.nn as nn
import torch.nn.functional as F
import math
  
    
class Net(nn.Module):
    def __init__(self, features, task, invasive, multi, check='none', out='none', ehr_dim = None,use_ehr = False):
        super(Net, self).__init__()   
        self.task, self.invasive, self.multi = task, invasive, multi
        self.out = out
        self.use_ehr = use_ehr

        if features == 'ecg_time' or features == 'co2_time' or features == 'ecg_freq' or features == 'co2_freq':
            self.inc = 1
        elif self.multi == True:
            self.inc = 4 if self.invasive == True else 3
        else:
            self.inc = 1
        if check == "fourier":
            self.inc = 2*self.inc
            
        if use_ehr == False:
            self.ehr_dim = 0
        else:
            self.ehr_dim = ehr_dim
                        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(0.3)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(0.3)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(0.3)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(0.3)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(320 + self.ehr_dim, 2),
            nn.Dropout(0.3)
        )
        
        self.activation = nn.Sigmoid()

    
    def forward(self, x, ehr=None):
#         import pdb; pdb.set_trace()
        x = x.view(x.shape[0], self.inc, -1)
    
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        if self.out != 'none':
            return out

        if self.use_ehr ==True:
            out = torch.cat([out,ehr], dim =1)
            
        out = self.fc(out)
        if self.task == 'classification':
            out = self.activation(out)
        
        return out

class Nets(nn.Module):
    def __init__(self, features, window, opt, task, invasive, multi, check='none', ehr_dim = None, use_ehr = False):
        super(Nets, self).__init__()   
        self.task, self.invasive, self.multi = task, invasive, multi

        self.time = Net( features = features, task = 'classification', invasive = opt['invasive'], 
                        multi = opt['multi'], out= "comb")
        self.freq = Net_freq( features = features, window = window, task = 'classification', 
                             invasive = opt['invasive'], multi = opt['multi'], check = 'fourier', out= "comb")
        self.use_ehr = use_ehr
        if features == 'ecg_time' or features == 'co2_time' or features == 'ecg_freq' or features == 'co2_freq':
            self.inc = 2
            self.out = 640
        elif self.multi == True:
            self.inc = 8 if invasive == True else 6
#             self.out = 2880 if invasive == True else 640
            self.out = 640
        else:
            self.inc = 2
            self.out = 640
        
        if use_ehr == False:
            self.ehr_dim = 0
        else:
            self.ehr_dim = ehr_dim
            
        self.fc = nn.Sequential(
            nn.Linear(self.out + self.ehr_dim, 2),
            nn.Dropout(0.3)
        )
        
        self.activation = nn.Sigmoid()
    
    def forward(self, x, ehr=None):
#         import pdb; pdb.set_trace()
        x = x.view(x.shape[0], self.inc, -1)
        time = self.time(x[:,0:int(self.inc/2),:])
        freq = self.freq(x[:,int(self.inc/2):self.inc,:])
        out = torch.cat([time,freq], dim =1)
#         import pdb; pdb.set_trace()
        if self.use_ehr ==True:
            out = torch.cat([out,ehr], dim =1)
        out = self.fc(out)    
        return self.activation(out)