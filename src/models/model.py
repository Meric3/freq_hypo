import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# import xgboost as xgb
  
    
class Net(nn.Module):
    def __init__(self, features, window , opt, task, invasive, multi, check='none', out='none', ehr_dim = None,use_ehr = False):
        super(Net, self).__init__()   
        self.opt = opt 
        self.task, self.invasive, self.multi = task, invasive, multi
        self.out = out
        self.use_ehr = use_ehr
        self.window = window

        if features == 'ecg_time' or features == 'co2_time' or features == 'ecg_freq' or features == 'co2_freq':
            self.inc = 1
        elif self.multi == True:
            self.inc = 4 if self.invasive == True else 3
        else:
            self.inc = 1
        # if check == "fourier":
        #     self.inc = 2*self.inc
            
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
        self.conv8 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(0.3)
        )
        if self.window == 3000:
            len_ = 320
        elif self.window == 1000:
            len_ = 64
        elif self.window == 2000 and opt['model'] == '1d_cnn':
            len_ = 124
        elif self.window == 4000 and opt['model'] == '1d_cnn':
            len_ = 64
        elif self.window == 5000 and opt['model'] == '1d_cnn':
            len_ = 192
        elif self.window == 6000 and opt['model'] == '1d_cnn':
            len_ = 320
        else:
            len_ = 320
        self.fc = nn.Sequential(
            nn.Linear(len_ + self.ehr_dim, 2),
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
        if self.window > 1001:
            out = self.conv7(out)
        if self.window > 3001:
            out = self.conv8(out)
        
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        if self.out != 'none':
            return out

        if self.use_ehr ==True:
            out = torch.cat([out,ehr], dim =1)
            
        out = self.fc(out)
        if self.task == 'classification':
            out = self.activation(out)
        
        return out
class Net_freq(nn.Module):
    def __init__(self, features, window ,opt, task, invasive, multi, check='none', out='none', ehr_dim = None, use_ehr = False):
        super(Net_freq, self).__init__()   
        self.task, self.invasive, self.multi = task, invasive, multi
        self.out = out
        self.use_ehr = use_ehr
        
        if features == 'ecg_time' or features == 'co2_time' or features ==  'ecg_freq' or features == 'co2_freq':
            self.inc = 1
        elif self.multi == True :
            self.inc = 4 if self.invasive == True else 3
        else:
            self.inc = 1
        
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
            nn.Linear(320 + self.ehr_dim, self.ehr_dim),
            nn.Dropout(0.3)
        )
        
        self.activation = nn.Sigmoid()

    
    def forward(self, x, ehr=None):
        
        x = x.view(x.shape[0], self.inc, -1)
    
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        if self.out == 'comb':
            return out

        out = self.fc(out)
        out = self.activation(out)
        
        return out
class Nets(nn.Module):
    def __init__(self, features, window, opt, task, invasive, multi, check='none', ehr_dim = None, use_ehr = False):
        super(Nets, self).__init__()   
        self.window = window
        self.task, self.invasive, self.multi = task, invasive, multi

        self.time = Net( features = features, window = opt['window'], opt = opt, task = 'classification', invasive = opt['invasive'], 
                        multi = opt['multi'],check="comb", out= "comb")
        self.freq = Net( features = features, window = opt['window'], opt = opt, task = 'classification', 
                             invasive = opt['invasive'], multi = opt['multi'], check = 'comb', out= "comb")
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

        if self.window == 1000:
            # 132 -4 
            self.out = 128
        elif self.window == 2000:
            self.out = 128
        elif self.window == 4000:
            self.out = 128
        elif self.window == 5000:
            self.out = 384
        elif self.window == 6000:
            self.out = 640           
        self.fc = nn.Sequential(
            nn.Linear(self.out + self.ehr_dim, 2),
            nn.Dropout(0.3)
        )
        
        self.activation = nn.Sigmoid()
    
    def forward(self, x, ehr=None):
        x = x.view(x.shape[0], self.inc, -1)
        time = self.time(x[:,0:int(self.inc/2),:])
        freq = self.freq(x[:,int(self.inc/2):self.inc,:])
        out = torch.cat([time,freq], dim =1)

        if self.use_ehr ==True:
            out = torch.cat([out,ehr], dim =1)

        out = self.fc(out)    
        return self.activation(out)
    
class KNN():
    # tr = train, v = valid, t = test, d = data, l = label
    def setData(self,tr_d,tr_l,v_d,v_l,t_d):
        self.tr_d, self.v_d, self.t_d = tr_d, v_d, t_d
        self.tr_l, self.v_l = tr_l, v_l
    
    def predict_output(self):
        k_list = range(1,101)
        valid_auroc = []
        for k in k_list:
            classifier = KNeighborsClassifier(n_neighbors = k)
#             import pdb; pdb.set_trace()
            classifier.fit(self.tr_d, self.tr_l)
            predict = classifier.predict(self.v_d)
            valid_auroc.append(roc_auc_score(predict,self.v_l))

        k = k_list[valid_auroc.index(max(valid_auroc))]

        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(self.tr_d,self.tr_l)
        predict = classifier.predict(self.t_d)

        return predict

class SVM():
    
    def setData(self,tr_d,tr_l,v_d,v_l,t_d):
        self.tr_d, self.v_d, self.t_d = tr_d, v_d, t_d
        self.tr_l, self.v_l = tr_l, v_l
    
    def predict_output(self):

        clf = svm.SVC(kernel='linear')
        clf.fit(self.tr_d, self.tr_l)
        predict = clf.predict(self.t_d)

        return predict

class RF():

    def setData(self,tr_d,tr_l,v_d,v_l,t_d):
        self.tr_d, self.v_d, self.t_d = tr_d, v_d, t_d
        self.tr_l, self.v_l = tr_l, v_l

    def predict_output(self):
        e,d = 200,10
        clf = RandomForestClassifier(n_estimators=e, max_depth=d,random_state=0)
        clf.fit(self.tr_d, self.tr_l)
        predict = clf.predict(self.t_d)

        return predict    
    
class XGBoost():
    
    def setData(self,tr_d,tr_l,v_d,v_l,t_d):
        self.tr_d, self.v_d, self.t_d = tr_d, v_d, t_d
        self.tr_l, self.v_l = tr_l, v_l
    
    def predict_output(self):

        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        xgb_model.fit(self.tr_d, self.tr_l)
        predict = xgb_model.predict(self.t_d)

        return predict    
    
class DNN_time(nn.Module):
    def __init__(self, features, task, invasive, multi, check='none', out='none', ehr_dim = None,use_ehr = False):
        super(DNN_time, self).__init__()   
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
            
            
            
        self.inc = self.inc * 3000
            

            
        self.dnn = nn.Sequential(
            nn.Linear(self.inc + self.ehr_dim, 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
            
        self.fc = nn.Sequential(
            nn.Linear(256, 2),
            nn.Dropout(0.3)
        )
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x, ehr=None):
#         import pdb; pdb.set_trace()
        x = x.view(x.shape[0], -1)
        x = torch.cat([x,ehr], dim =1)
        out = self.dnn(x)

        if self.out != 'none':
            return out
        out = self.fc(out)
        
        if self.task == 'classification':
            out = self.activation(out)
        
        return out
    
    
class DNNs(nn.Module):
    def __init__(self, features, window, opt, task, invasive, multi, check='none', ehr_dim = None, use_ehr = False):
        super(DNNs, self).__init__()   
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
        if self.use_ehr ==True and ehr != None :
            out = torch.cat([out, ehr], dim =1)
        elif self.use_ehr ==True and ehr == None:
#             import pdb; pdb.set_trace()
#             out = torch.cat([out, ehr], dim =1)
            out = torch.cat([out, torch.zeros_like(out)[:,:4]], dim =1)
#             print("@@@@")
        out = self.fc(out)    
        return self.activation(out)
    
    
    
#     LSTM
#     LSTM
class LSTM_time(nn.Module):
    def __init__(self, features, task, invasive, multi, check='none', out='none', ehr_dim = None,use_ehr = False):
        super(LSTM_time, self).__init__()   
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
        
        self.lstm = nn.LSTM(self.inc, 64, batch_first=True)
        self.label = nn.Linear(self.inc + self.ehr_dim, 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fc = nn.Sequential(
            nn.Linear(64 + self.ehr_dim, 2),
            nn.Dropout(0.3)
        )
        self.activation = nn.Sigmoid()

        
    def forward(self, x, ehr=None):
#         import pdb; pdb.set_trace()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.randn(1, x.shape[0], 64).to(self.device)
        c0 = torch.randn(1, x.shape[0], 64).to(self.device)
        out, (hn, cn) = self.lstm(x.permute(0,2,1), (h0, c0))
        
#         out = torch.cat([out,ehr], dim =1)
        out = torch.cat([hn[-1], ehr], dim =1)
        
#         hn[-1].shape
#         pred = fully_connect_layer(hidden_state[-1])
#         x = x.view(x.shape[0], -1)
#         x = torch.cat([x,ehr], dim =1)
#         out = self.dnn(x)

        if self.out != 'none':
            return out
        out = self.fc(out)
        
        if self.task == 'classification':
            out = self.activation(out)
        
        return out
    


# http://202.30.23.33:9103/notebooks/home/mjh319/Untitled.ipynb
class VAE(nn.Module):
    def __init__(self, latent_dim, features, window, 
                 opt, task, invasive, multi, check='none', 
                 ehr_dim = None, use_ehr = False):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.sizes = 3000
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Linear(self.sizes , 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # 잠재 변수의 평균과 로그 분산
        )

        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.sizes ),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, ehr=None):
        # 인코더를 통해 평균과 로그 분산 계산
        latent = self.encoder(x)
        mu, logvar = latent[:, :self.latent_dim], latent[:, self.latent_dim:]
        
        # 잠재 공간에서 샘플링
        z = self.reparameterize(mu, logvar)
        
        # 디코더로 복원
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    










class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z

class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w

class UsadModel(nn.Module):
  def __init__(self, w_size, z_size, features, window, 
                 opt, task, invasive, multi, check='none', 
                 ehr_dim = None, use_ehr = False):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))