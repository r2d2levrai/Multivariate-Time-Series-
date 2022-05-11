""" Attention Input shape --> [Batch , seq,..]"""

import torch
import torch.nn as nn
import math
class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device, bidirectional=False):
        super(TimeLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.U_all = torch.nn.Parameter(torch.Tensor( input_size,hidden_size * 4))
        self.W_d = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bidirectional = bidirectional
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, timestamps, hidden_states=None, reverse=False):
        b, seq, embed = inputs.size()
        if hidden_states is None:
          h = torch.zeros(b, self.hidden_size, requires_grad=False)
          _c = torch.zeros(b, self.hidden_size, requires_grad=False)
        else:
          h,_c = hidden_states

        outputs = []
        hidden_state_h = []
        hidden_state_c = []


        for s in range(seq):
            c_s1 = torch.tanh(_c@self.W_d) # short term mem
            #print("shape C_s1",c_s1.shape)
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1) # discounted short term mem
            c_l = -c_s1+ _c
            c_adj =c_l+ c_s2
            
            W_f, W_i, W_o, W_c_tmp = self.W_all.chunk(4, dim=1)
            U_f, U_i, U_o, U_c_tmp = self.U_all.chunk(4, dim=1)

            f = torch.sigmoid(h@W_f+inputs[:, s]@U_f)

            i =torch.sigmoid(h@W_i+inputs[:, s]@U_i)
            o =torch.sigmoid(h@W_o+inputs[:, s]@U_o)
            c_tmp = torch.sigmoid(h@W_c_tmp+inputs[:, s]@U_c_tmp)
            
            _c = f*c_adj+i*c_tmp
            h=o*torch.tanh(_c)

            outputs.append(o)
            hidden_state_c.append(_c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()
        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, _c)
