import torch
import torch.nn as nn
from torch.autograd import Variable
# from sru import SRU, SRUCell

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MGU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):
        batch_size, path_length, emb_dim = input.size()

        input = input.view(-1, emb_dim)
        input = self.Wx(input)
        input = input.view(batch_size, path_length, self.hidden_size)

        h = torch.zeros(batch_size, self.hidden_size).cuda()

        outputs = []
        for i in range(path_length):
            x = input[:, i, :]
            x = torch.sigmoid(x)
            h = h * x + (1 - x) * torch.tanh(self.Wh(h))
            outputs.append(h)

        return torch.stack(outputs, dim=1)




 


