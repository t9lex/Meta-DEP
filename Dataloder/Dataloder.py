import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from .Dataset import PathDataset

# train dataset
with open("process_data/path_dic_train.pkl", 'rb') as f: 
    path_dict_train = pickle.load(f)

# test dataset
with open("process_data/path_dic_recall.pkl", 'rb') as f: 
    path_dict_test = pickle.load(f)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
rng = np.random.RandomState(0)

print('Creating tensor dataset...') 
drug_disease_array_train = list(path_dict_train.keys())
drug_disease_array_test = list(path_dict_test.keys())

dataset_train = PathDataset(drug_disease_array=drug_disease_array_train,
                              total_path_dict=path_dict_train,
                              max_path_length=8,
                              max_path_num=64,
                              rng=rng)

dataset_test = PathDataset(drug_disease_array=drug_disease_array_test,
                                total_path_dict=path_dict_test,
                                max_path_length=8,
                                max_path_num=64,
                                rng=rng)


train_data_loader = DataLoader(dataset=dataset_train, batch_size=5, shuffle=True, num_workers=2,drop_last=False)
test_data_loader = DataLoader(dataset=dataset_test, batch_size= 99 ,shuffle=False, num_workers=0,drop_last=True)






