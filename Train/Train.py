import torch
import numpy as np
import pickle
import torch
import torch.nn as nn
from Model.pytorchtools import EarlyStopping 
import Model.loss as module_loss
import Model.metric as module_metric
from Model.Meta_DES import Meta_DES
from Trainer.Trainer import train_epoch
from Trainer.Trainer import test_epoch

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#embedding
with open("process_data/metapath2vec_emb.pkl","rb") as f:
    ppi_emd = pickle.load(f)
ppi_emd = [emd.tolist() for emd in ppi_emd]
ppi_emd
ppi_emd = torch.tensor(ppi_emd).to(device) 

#model
model = Meta_DES(first_dim=128,emb_dim=128)
model.to(device) 
  
# loss
#criterion_softmax = getattr(module_loss,  "cross_entropy_loss")
criterion = getattr(module_loss,  "mse_loss")
criterion_softmax = nn.CrossEntropyLoss()

# # metrics
metrics = [getattr(module_metric, met) for met in ["mae", "mse","rmse", "r2","pearson"]]
metrics_ = [getattr(module_metric, met) for met in ["accuracy","recall","precision","f1_score"]]

# build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=1e-5,amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 30, 50, 100], gamma=0.1)

# earlystopping
early_stopping = EarlyStopping(patience=30, verbose=False, path='cheackpoint/checkpoint_train.pt')


best_log = {'mae': 0,"mse":0,"rmse":float("inf"), "r2": 0,"pearson": 0,"accuracy":0,"recall":0,"precision":0,"f1_score":0}
for epoch in range(1, 151):
    print('Epoch: {}'.format(epoch))
    log_train = train_epoch()
    test_log,out_list,output = test_epoch()
    print(test_log)
    # print(log_train)   
    performance = log_train["rmse"]
    print(best_log)
    scheduler.step()
    if log_train["rmse"] <  best_log['rmse']:
        best_log.update(mae = log_train['mae'], mse = log_train['mse'], rmse = log_train['rmse'], r2 = log_train['r2'], pearson = log_train['pearson'], accuracy = log_train['accuracy'],  recall=log_train["recall"],precision=log_train["precision"],f1_score=log_train["f1_score"])       
    early_stopping(performance, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
print('Best performance: {}'.format(best_log))












