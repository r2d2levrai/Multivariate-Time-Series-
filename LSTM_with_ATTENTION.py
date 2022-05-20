import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class TPA_LSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False,
                 attention_width=3, cuda=False):
        super(TPA_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(ntoken, ninp)
        
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.softmax = nn.Softmax()
        
        self.AttentionLayer = AttentionLayer(cuda,nhid)
        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers
        self.attention_width = attention_width

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden,cell):
        #print("input size:",input.size())
        emb = self.drop(self.encoder(input))
        #print("emb size:",emb.size())
        hidden=(hidden,cell)
        output, hidden = self.rnn(emb,hidden)
        #print("rnn output",output.size())
        #print("rnn hidden",hidden[0].size())
        
        output = self.AttentionLayer.forward(output, self.attention_width)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

class AttentionLayer(nn.Module):
    """Implements an Attention Layer"""

    def __init__(self, cuda, nhid):
        super(AttentionLayer, self).__init__()
        self.nhid = nhid
        self.weight_W = nn.Parameter(torch.Tensor(nhid,nhid))
        self.weight_proj = nn.Parameter(torch.Tensor(nhid, 1))
        self.softmax = nn.Softmax()
        self.weight_W.data.uniform_(-0.1, 0.1)
        self.weight_proj.data.uniform_(-0.1,0.1)
        self.cuda = cuda

    def forward(self, inputs, attention_width=3):
        results = None
        for i in range(inputs.size(0)):
            if(i<attention_width):
                output = inputs[i]
                output = output.unsqueeze(0)

            else:
                lb = i - attention_width
                if(lb<0):
                    lb = 0
                selector = torch.from_numpy(np.array(np.arange(lb, i)))
                if self.cuda:
                    selector = Variable(selector).cuda()
                else:
                    selector = Variable(selector)
                vec = torch.index_select(inputs, 0, selector)
                u = batch_matmul(vec, self.weight_W, nonlinearity='tanh')
                a = batch_matmul(u, self.weight_proj)
                a = self.softmax(a)
                output = None
                for i in range(vec.size(0)):
                    h_i = vec[i]
                    a_i = a[i].unsqueeze(1).expand_as(h_i)
                    h_i = a_i * h_i
                    h_i = h_i.unsqueeze(0)
                    if(output is None):
                        output = h_i
                    else:
                        output = torch.cat((output,h_i),0)
                output = torch.sum(output,0)
                output = output.unsqueeze(0)

            if(results is None):
                results = output

            else:
                results = torch.cat((results,output),0)

        return results
    
def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()
