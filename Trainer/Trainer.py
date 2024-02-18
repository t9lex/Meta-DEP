import numpy as np
import pickle
import torch
from model.pytorchtools import EarlyStopping 
import model.loss as module_loss
import Model.metric as module_metric
from Model import Meta_DES
from Dataloder.Dataloder import train_data_loader
from Dataloder.Dataloder import test_data_loader

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#embedding
with open("metapath2vec_emb.pkl","rb") as f:
    ppi_emd = pickle.load(f)
ppi_emd = [emd.tolist() for emd in ppi_emd]
ppi_emd 

#model
model = Meta_DES(first_dim=128,emb_dim=128)
model.to(device) 
  
# loss
criterion_softmax = getattr(module_loss,  "cross_entropy_loss")
criterion = getattr(module_loss,  "mse_loss")

# metrics
metrics = [getattr(module_metric, met) for met in ["mae", "mse","rmse", "r2","pearson","accuracy","recall","precision","f1_score"]]
metrics_ = [getattr(module_metric, met) for met in ["accuracy","recall","precision","f1_score"]]

# build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=1e-5,amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 30, 50, 100], gamma=0.1)

# earlystopping
early_stopping = EarlyStopping(patience=30, verbose=False, path='../checkpoint/checkpoint_train.pt')

def train_epoch():
        model.train()

        pre_list = []
        true_list = []

        pre_score_list = []
        true_score_list = []

        log = {'mae': 0,"mse":0,"rmse":0, "r2": 0,"pearson": 0,"accuracy":0,"recall":0,"precision":0,"f1_score":0}
        for batch_idx, (_, _, path_feature, lengths, mask, target,lable_true) in enumerate(train_data_loader):
            path_feature = path_feature.to(device)
            mask, target,lable_true = mask.to(device), target.to(device),lable_true.to(device)
            optimizer.zero_grad()
            output,out_sigmod,out_softmax, _, _ = model(path_feature, lengths, mask,ppi_emd)
            
            pre = [1 if item[0] > 0.065 else 0 for item in output]
            pre_list.extend(pre)
            true = lable_true.cpu().detach().numpy()
            true_list.extend(true)

            loss_ = criterion(output, target)
            loss_softmax = criterion_softmax(out_softmax, torch.squeeze(lable_true).long())
            loss = loss_ + 0.3*loss_softmax
            loss.backward()
            optimizer.step()
            log.update(loss = loss.item())
            print('loss: {:.4f}'.format(loss.item()))
            with torch.no_grad():
                y_pred = output.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                pre_score_list.extend(y_pred)
                true_score_list.extend(y_true)

        pre_score = np.array(pre_score_list)
        true_score = np.array(true_score_list)
        metrics_list = [met(pre_score, true_score) for met in metrics]
        metrics_list_ = [met(pre_list,true_list) for met in metrics_] 
        log.update(mae = metrics_list[0], mse = metrics_list[1], rmse =metrics_list[2], r2 = metrics_list[3], pearson = metrics_list[4], accuracy =metrics_list_[0],  recall = metrics_list_[1], precision = metrics_list_[2], f1_score = metrics_list_[3])
        return log

def test_epoch():
        model.eval()

        pre_list = []
        true_list = []

        pre_score_list = []
        true_score_list = []

        out_list = []

        log = {'mae': 0,"mse":0,"rmse":0, "r2": 0,"pearson": 0,"accuracy":0,"recall":0,"precision":0,"f1_score":0}
        for batch_idx, (_, _, path_feature, lengths, mask, target,lable_true) in enumerate(test_data_loader):
            path_feature = path_feature.to(device)
            mask, target,lable_true = mask.to(device), target.to(device), lable_true.to(device)
            output,out_sigmod,out_softmax, _, _ = model(path_feature, lengths, mask,ppi_emd)
                
            for item in output:
                out_list.append(float(item[0].cpu().detach()))

            pre = [1 if item[0] > 0.065 else 0 for item in output]
            pre_list.extend(pre)
            true = lable_true.cpu().detach().numpy()
            true_list.extend(true)
            
            with torch.no_grad():
                y_pred = output.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                pre_score_list.extend(y_pred)
                true_score_list.extend(y_true)

        pre_score = np.array(pre_score_list)
        true_score = np.array(true_score_list)
        metrics_list = [met(pre_score, true_score) for met in metrics]
        metrics_list_ = [met(pre_list,true_list) for met in metrics_] 
        log.update(mae = metrics_list[0], mse = metrics_list[1], rmse =metrics_list[2], r2 = metrics_list[3], pearson = metrics_list[4], accuracy =metrics_list_[0],  recall = metrics_list_[1], precision = metrics_list_[2], f1_score = metrics_list_[3])
        return log , out_list,output

def predict():
    model.eval()
    for batch_idx, (_, _, path_feature, lengths, mask, target,lable_true) in enumerate(test_data_loader):
        path_feature = path_feature.to(device)
        mask, target,lable_true = mask.to(device), target.to(device), lable_true.to(device)
        output,out_sigmod,out_softmax, _, _ = model(path_feature, lengths, mask,ppi_emd)
        return output








