import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import Dataset
import numpy as np

class Meta_DES(nn.Module):
    def __init__(self, first_dim,emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.liner = nn.Linear(in_features=first_dim, out_features=emb_dim)

        self.RNN = nn.RNN(input_size=emb_dim, hidden_size=emb_dim)

        self.node_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.node_attention_softmax = nn.Softmax(dim=1)

        self.path_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.path_attention_softmax = nn.Softmax(dim=1)

        self.hidden_linear = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.output_linear = nn.Linear(in_features=emb_dim, out_features=1)

        #softmax
        self.output_linear_new = nn.Linear(in_features=emb_dim+1, out_features=2)       
        self.softmax = nn.Softmax(dim=1)
        #sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, path_feature, lengths, mask, ppi_emd):
        # shape of path_feature: [batch_size, path_num, path_length]
        # shape of type_feature: [batch_size, path_num, path_length]
        '''Metapath2vc embedding'''
        value_embedding = ppi_emd

        '''Embedding'''
        batch, path_num, path_len = path_feature.size()
        path_feature = path_feature.view(batch*path_num, path_len)
        # shape of path_embedding: [batch_size*path_num, path_length, emb_dim]
        path_embedding = value_embedding[path_feature]
        feature = path_embedding

        '''Linear'''
        # shape of feature: [batch_size*path_num, path_length, emb_dim]
        feature = self.liner(feature)

        '''Pack padded sequence'''
        feature = torch.transpose(feature, dim0=0, dim1=1)
        feature = utils.rnn.pack_padded_sequence(feature, lengths=list(lengths.view(batch*path_num).data),
                                                 enforce_sorted=False)
        
        '''RNN'''
        # shape of lstm_out: [path_length, batch_size*path_num, emb_dim]
        RNN_out, _ = self.RNN(feature)
        # unpack, shape of lstm_out: [batch_size*path_num, path_length, emb_dim]
        RNN_out, _ = utils.rnn.pad_packed_sequence(RNN_out, batch_first=True, total_length=path_len)
        
        '''Node attention'''
        # shape of output_path_embedding: [batch_size*path_num, emb_dim]
        mask = mask.view(batch*path_num, path_len)
        output_path_embedding, node_weight_normalized = self.node_attention(RNN_out, mask)
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









































