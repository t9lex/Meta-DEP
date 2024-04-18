#  -*-coding:utf8 -*-
import os
import sys
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import itertools
import torch
import torch.nn as nn
import torch.nn.utils as utils
import model.metric as module_metric
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from model.Metapath2vec import MetaPath2Vec as MetaPath2Vec
###########################################

def process_txt_file(file_path):
    """
    读取指定.txt文件，将每行内容存储到一个列表中，并将列表以文件名（去掉扩展名）为键存入一个字典中。
    
    参数:
    file_path (str): 待处理的.txt文件路径
    
    返回:
    dict, list: 包含处理后数据的字典和列表
    """
    ESCC_gen = {}
    ESCC_list = []

    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 打开文本文件
    with open(file_path, 'r') as f:
        # 遍历每一行
        for line in f:
            # 处理每一行
            ESCC_list.append(line.strip())

    # 将列表以文件名（去掉扩展名）为键存入字典
    ESCC_gen[base_name] = ESCC_list

    return ESCC_gen, ESCC_list

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_txt.py <file_name>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dict, output_list = process_txt_file(file_path)

    print(f"Length of the disease protein list: {len(output_list)}")

ESCC_gen,ESCC_list = process_txt_file(file_path)


print("################################################################# Step : 1 Metapath2vec ##################################################################")
print("####################################################This step will take some time, please be patient######################################################")
with open("gene_name2id.pkl","rb") as g:
    new_dict = pickle.load(g)

dict_map = np.load("dict_map_d_ppi_d.npy",allow_pickle=True).item()

escc_new = []
for gene in ESCC_list:
    if gene in new_dict.keys():
        if new_dict[gene] in dict_map.keys():
            escc_new.append(new_dict[gene])
print("Length of processed disease protein list",len(escc_new))

#ppi
f =open("intercome_new.lcc","r")

#disease_drug_pair
disease_drug = np.load("disease_drug_new.npy",allow_pickle=True)  # 读取疾病与药物的映射关系

#disease_gene_pair
with open("disease_gen.pkl","rb") as ds:
    disease_gen = pickle.load(ds)
# you disease_gene_pair
disease_gen["disease_new"] = escc_new

#drug_gene_pair
with open("drug_gen.pkl","rb") as d:
    drug_gen = pickle.load(d)

G=nx.Graph()
for line in f:
    head=line.split()[0]
    tail=line.split()[2]
    G.add_edge(head,tail)

n_di = 0
disease_map = {}
for disease in disease_gen.keys():
    disease_map[disease] = n_di
    n_di+= 1

n_dr = 0
drug_map = {}
for drug in drug_gen.keys():
    drug_map[drug] = n_dr
    n_dr += 1

n_gene = 0
gene_map = {}
for gen in G.nodes():
    gene_map[gen] = n_gene
    n_gene += 1

data = HeteroData()

# disease_gen_list = []
gen_dis_list = []
for disease,gen in disease_gen.items():
    for gene in gen:
        gen_dis_list.append([gene_map[gene],disease_map[disease]])
    
drug_gen_list = []
for drug,gen in drug_gen.items():
    for gene in gen:
        drug_gen_list.append([drug_map[drug],gene_map[gene]])

gene_gene_list = []
for gen in G.nodes():
    for gene in G.neighbors(gen):
        gene_gene_list.append([gene_map[gen],gene_map[gene]])

drug_dis_list = []
for disease,drug in disease_drug:
    drug_dis_list.append([drug_map[drug],disease_map[disease]])


data['gene', 'gene'].edge_index = torch.tensor(gene_gene_list).t().contiguous()
data['gene',  'disease'].edge_index = torch.tensor(gen_dis_list).t().contiguous()
data["drug","disease"].edge_index = torch.tensor(drug_dis_list).t().contiguous()
data['drug',  'gene'].edge_index = torch.tensor(drug_gen_list).t().contiguous()


print(data)
print(data.edge_index_dict)
print(data.edge_types)

metapath = [("drug","to","gene"),("gene","to","gene"),("gene","to","disease")]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = MetaPath2Vec(edge_index_dict=data.edge_index_dict, # 边类型字典
                     embedding_dim=128, # 节点维度嵌入长度
                     metapath = metapath, # 元路径
                     walk_length=1, # 序列游走长度
                     context_size=2, # 上下文大小
                     walks_per_node=1, # 每个节点游走10个序列
                     num_negative_samples=1, # 负采样数量
                    #  num_nodes_dict= data.num_nodes_dict
                    sparse=True # 权重设置为稀疏矩阵
                    ).to(device)

print(model.num_nodes_dict)

loader = model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

loss_stats = 10000
for epoch in range(1, 500): #7000
    loss = train()
    # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    if loss < loss_stats:
        loss_stats = loss
        torch.save(model.state_dict(), './model/metapath2vec.pt')
    # scheduler.step()
model.load_state_dict(torch.load('./model/metapath2vec.pt'))
model.eval()    
emb_dis = model("disease")
emb_drug = model("drug")
emb_gene = model("gene")

#save to cpu and numpy
emb_dis = emb_dis.cpu().detach().numpy()
emb_drug = emb_drug.cpu().detach().numpy()
emb_gene = emb_gene.cpu().detach().numpy()

dic_metapath = {}
for disease,num in disease_map.items():
    # print(disease,emb_dis[num])
    dic_metapath[disease] = emb_dis[num]
for drug,num in drug_map.items():
    # print(drug,emb_drug[num])
    dic_metapath[drug] = emb_drug[num]
for gene,num in gene_map.items():
    # print(gene,emb_gene[num])
    dic_metapath[gene] = emb_gene[num]

#save the list
with open('New_disease_metapath2vec_emb.pkl', 'wb') as f:
    pickle.dump(dic_metapath, f)


print("#####################################################   Step 2 : Find shortest paths   ###################################################################")
print("####################################################This step will take some time, please be patient######################################################")
dict_map = np.load("dict_map_d_ppi_d.npy",allow_pickle=True).item()
dict_map["disease_new"] = 19124

with open("gene_name2id.pkl","rb") as g:
    new_dict = pickle.load(g)

#################读取中药单体基因###############
with open("danti2tar.pkl","rb") as d:
    # dic_danti_gen = pickle.load(d)
    dic_danti = pickle.load(d)
key_value_pairs = itertools.islice(dic_danti.items(), 100)
dic_danti_gen = dict(key_value_pairs)

##############单体中药路径##################
n = open("intercome_new.lcc","r")

G=nx.Graph()
for line in n:
        head=line.split()[0]
        tail=line.split()[2]
        G.add_edge(dict_map[head],dict_map[tail])
Gcc = sorted(nx.connected_components(G),key = len,reverse=True)
largest_cc = G.subgraph(Gcc[0])
LCC = largest_cc.copy()
print(LCC)

path_dic_disease_danti = {}
T = LCC.copy()

for gene in ESCC_list:
    if gene in new_dict.keys():
        if new_dict[gene] in dict_map.keys():
            T.add_edge(dict_map["disease_new"],dict_map[new_dict[gene]])
for danti in dic_danti_gen.keys():   
    path_list = [] 
    for gen in dic_danti_gen[danti]:
        if gen in new_dict.keys():
            if new_dict[gen] in dict_map.keys(): 
                for path in nx.all_shortest_paths(T,dict_map[new_dict[gen]],dict_map["disease_new"]):
                    path_list.append(path)
    if len(path_list) != 0:
        path_dic_disease_danti[danti,"disease_new",1,1] = path_list

#save path_dic_disease_danti
with open('disease_danti_path.pickle','wb') as w:
    pickle.dump(path_dic_disease_danti,w)


print("##########################################################   Step 3: predict   #######################################################################")
with open("New_disease_metapath2vec_emb.pkl","rb") as f:
    emd_metapath2vec = pickle.load(f)

temp = []
for key in dict_map.keys():
    temp.append(emd_metapath2vec[key])

with open("New_disease_metapath2vec_emb_.pkl","wb") as f:
    pickle.dump(temp,f)

with open("New_disease_metapath2vec_emb_.pkl","rb") as f:
    ppi_emd = pickle.load(f)
ppi_emd = [emd.tolist() for emd in ppi_emd]


with open('disease_danti_path.pickle','rb') as r:
    path_dict_neg = pickle.load(r)
danti_list =[]
for item in path_dict_neg.keys():
    danti_list.append(item[0])
print(danti_list)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
rng = np.random.RandomState(0)

print('Creating tensor dataset...') 
drug_disease_array_neg = list(path_dict_neg.keys())

class PathDataset(Dataset):
    def __init__(self, drug_disease_array, total_path_dict,  
                       max_path_length=8, max_path_num=256, rng=None):
        self.drug_disease_array = drug_disease_array
        self.total_path_dict = total_path_dict
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.rng = rng

    def __len__(self):
        return len(self.drug_disease_array)

    def __getitem__(self, index):
        drug, disease, label, lable_new = self.drug_disease_array[index]
        path_list = self.total_path_dict[tuple([drug, disease, label, lable_new])]
        path_array_list = []
        lengths_list = []
        mask_list = []
        for path in path_list:
            path = path[:self.max_path_length]
            pad_num = max(0, self.max_path_length - len(path))
            path_array_list.append(path + [0]*pad_num)
            lengths_list.append(len(path))
            mask_list.append([1]*len(path)+[0]*pad_num)
        replace = len(path_array_list) < self.max_path_num
        select_idx_list = [idx for idx in self.rng.choice(len(path_array_list), size=self.max_path_num, replace=replace)]
        path_array = np.array([path_array_list[idx] for idx in select_idx_list])
        lengths_array = np.array([lengths_list[idx] for idx in select_idx_list])
        mask_array = np.array([mask_list[idx] for idx in select_idx_list])

        path_feature = torch.from_numpy(path_array).type(torch.LongTensor)
        label = torch.from_numpy(np.array([label])).type(torch.FloatTensor)
        new_lable = torch.from_numpy(np.array([lable_new])).type(torch.FloatTensor)
        lengths = torch.from_numpy(lengths_array).type(torch.LongTensor)
        mask = torch.from_numpy(mask_array).type(torch.ByteTensor)

        return drug, disease, path_feature, lengths, mask, label, new_lable

dataset_neg = PathDataset(drug_disease_array=drug_disease_array_neg,
                                total_path_dict=path_dict_neg,
                                max_path_length=8,
                                max_path_num=64,
                                rng=rng)

class iDPath(nn.Module):
    def __init__(self, first_dim,emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.liner = nn.Linear(in_features=first_dim, out_features=emb_dim)

        self.lstm = nn.RNN(input_size=emb_dim, hidden_size=emb_dim)

        self.node_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.node_attention_softmax = nn.Softmax(dim=1)

        self.path_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.path_attention_softmax = nn.Softmax(dim=1)

        self.hidden_linear = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.output_linear = nn.Linear(in_features=emb_dim, out_features=1)

        #softmax用于二分类
        self.output_linear_new = nn.Linear(in_features=emb_dim+1, out_features=2)       
        self.softmax = nn.Softmax(dim=1)
        #把out_featrue做sigmoid用于二分类
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, path_feature, lengths, mask, ppi_emd):
        # shape of path_feature: [batch_size, path_num, path_length]
        # shape of type_feature: [batch_size, path_num, path_length]
        # '''GCN embedding'''

        '''GCN embedding'''
        gcn_value_embedding = torch.tensor(ppi_emd).to(device)
        # ego_value_embedding = [emd.tolist() for emd in ppi_emd]
        # gcn_value_embedding = torch.tensor(ego_value_embedding).to(device)

        '''Embedding'''
        batch, path_num, path_len = path_feature.size()
        path_feature = path_feature.view(batch*path_num, path_len)
        # shape of path_embedding: [batch_size*path_num, path_length, emb_dim]
        path_embedding = gcn_value_embedding[path_feature]
        feature = path_embedding

        '''Linear'''
        # shape of feature: [batch_size*path_num, path_length, emb_dim]
        feature = self.liner(feature)

        '''Pack padded sequence'''
        feature = torch.transpose(feature, dim0=0, dim1=1)
        feature = utils.rnn.pack_padded_sequence(feature, lengths=list(lengths.view(batch*path_num).data),
                                                 enforce_sorted=False)
        
        '''LSTM'''
        # shape of lstm_out: [path_length, batch_size*path_num, emb_dim]
        lstm_out, _ = self.lstm(feature)
        # unpack, shape of lstm_out: [batch_size*path_num, path_length, emb_dim]
        lstm_out, _ = utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=path_len)
        
        '''Node attention'''
        # shape of output_path_embedding: [batch_size*path_num, emb_dim]
        mask = mask.view(batch*path_num, path_len)
        output_path_embedding, node_weight_normalized = self.node_attention(lstm_out, mask)
        # the original shape of node_weight_normalized: [batch_size*path_num, path_length]
        node_weight_normalized = node_weight_normalized.view(batch, path_num, path_len)
        # shape of output_path_embedding: [batch_size, path_num, emb_dim]
        output_path_embedding = output_path_embedding.view(batch, path_num, self.emb_dim)
        
        '''Path attention'''
        # shape of output_path_embedding: [batch_size, emb_dim]
        # shape of path_weight_normalized: [batch_size, path_num]
        output_embedding, path_weight_normalized = self.path_attention(output_path_embedding)

                
        '''Prediction'''
        hidden_embedding = self.hidden_linear(output_embedding)
        # print(hidden_embedding.shape)
        output = self.output_linear(hidden_embedding)
        # print(output.shape)

        '''softmax'''
        softmax_embedding = torch.cat((hidden_embedding,output),dim=1)
        output_new = self.output_linear_new(softmax_embedding)
        out_softmax = self.softmax(output_new)

        '''sigmoid'''
        out_sigmod = self.sigmoid(output)
        
        return output, out_sigmod,out_softmax, node_weight_normalized, path_weight_normalized

    def node_attention(self, input, mask):
        # the shape of input: [batch_size*path_num, path_length, emb_dim]
        weight = self.node_attention_linear(input) # shape: [batch_size*path_num, path_length, 1]
        # shape: [batch_size*path_num, path_length]
        weight = weight.squeeze() 
        '''mask'''
        # the shape of mask: [batch_size*path_num, path_length]
        weight = weight.masked_fill(mask==0, torch.tensor(-1e9))
        # shape: [batch_size*path_num, path_length]
        weight_normalized = self.node_attention_softmax(weight) 
        # shape: [batch_size*path_num, path_length, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2) 
        # shape: [batch_size*path_num, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1) 
        return input_weighted, weight_normalized

    def path_attention(self, input):
        # the shape of input: [batch_size, path_num, emb_dim]
        weight = self.path_attention_linear(input)
        # [batch_size, path_num]
        weight = weight.squeeze()
        # [batch_size, path_num]
        weight_normalized = self.path_attention_softmax(weight)
        # [batch_size, path_num, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2)
        # [batch_size, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1)
        return input_weighted, weight_normalized

metrics = [getattr(module_metric, met) for met in ["mae", "mse","rmse", "r2","pearson"]]
metrics1 = [getattr(module_metric, met) for met in ["accuracy","recall","precision","f1_score"]]

test_data_loader = DataLoader(dataset=dataset_neg, batch_size= 99 ,shuffle=False, num_workers=0,drop_last=False)

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
            # print(output)
                
            for item in output:
                # out_list.append(math.log(math.pow(10,-item[0])))
                out_list.append(float(item[0].cpu().detach()))

            pre = [1 if item[0] > 0.065 else 0 for item in output]
            pre_list.extend(pre)
            # print(pre)
            true = lable_true.cpu().detach().numpy()
            true_list.extend(true)
            # print(true)
            
            with torch.no_grad():
                y_pred = output.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                pre_score_list.extend(y_pred)
                true_score_list.extend(y_true)

        pre_score = np.array(pre_score_list)
        true_score = np.array(true_score_list)
        metrics_list = [met(pre_score, true_score) for met in metrics]
        metrics_list1 = [met(pre_list,true_list) for met in metrics1] 
        log.update(mae = metrics_list[0], mse = metrics_list[1], rmse =metrics_list[2], r2 = metrics_list[3], pearson = metrics_list[4], accuracy =metrics_list1[0],  recall = metrics_list1[1], precision = metrics_list1[2], f1_score = metrics_list1[3])
        return log , out_list,output

model = iDPath(first_dim=128,emb_dim=128)
model.to(device) 
print(model)  

model.load_state_dict(torch.load('checkpoint/checkpoint.pt'))
test_log,out_list,output = test_epoch()

# monomer_score
dict_danti_score = {k:v for k,v in zip(danti_list,out_list)}
sorted_dict = dict(sorted(dict_danti_score.items(), key=lambda x:x[1], reverse=True))

#save sorted dict to csv
df = pd.DataFrame(sorted_dict.items(), columns=['monomer', 'score'])
df.to_csv('sorted_dict.csv', index=False)

print("########################################################################## DONE ##########################################################################")
print("################################################################The sorted_dict.csv is saved##############################################################")