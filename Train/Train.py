import torch
from Trainer.Trainer import train_epoch
from Trainer.Trainer import test_epoch

best_log = {'mae': 0,"mse":0,"rmse":float("inf"), "r2": 0,"pearson": 0,"accuracy":0,"recall":0,"precision":0,"f1_score":0}
for epoch in range(1, 151):
    print('Epoch: {}'.format(epoch))
    log_train = train_epoch()
    test_log,out_list = test_epoch()
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












