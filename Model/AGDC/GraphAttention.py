from numpy import double
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAtten(nn.Module):
    def __init__(self, n_head, d_in, nodes, droprate=0.1):
        super(GraphAtten, self).__init__()

        self.d_in = d_in
        self.n_head = n_head
        self.nodes = nodes
        # self.d_out = d_out
        self.droprate = droprate

        # self.fc = nn.Linear(d_in, d_out, bias=False)
        self.slf_att1 = nn.Conv1d(d_in, n_head, 1, bias=False)
        self.slf_att2 = nn.Conv1d(d_in, n_head, 1, bias=False)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(droprate)

        self.norm = nn.LayerNorm([nodes, nodes])

    def forward(self, x, adj, with_batch=True):
        # x = self.fc(x)

        # self-attention after optimization
        f_1 = self.slf_att1(torch.permute(x, [0, 2, 1]))  # shape size [sz_b, n_head, nodes]
        f_2 = self.slf_att2(torch.permute(x, [0, 2, 1]))  # shape size [sz_b, n_head, nodes]
        f_1 = torch.unsqueeze(f_1, dim=2)  # shape size [sz_b, n_head, 1, nodes]
        f_2 = torch.unsqueeze(f_2, dim=2)  # shape size [sz_b, n_head, 1, nodes]
        x = f_1 + torch.permute(f_2, [0, 1, 3, 2])  # shape size [sz_b, n_head, nodes, nodes]

        # x = self._prepare_attentional_mechanism_input(x)

        e = self.activation(torch.mean(x, dim=1))  # shape size [sz_b, nodes, nodes]
        # e = self.torch.mean(self.activation(self.att(x)), dim=-1)
        
        if not with_batch:
            e = torch.mean(e, dim=0)

        zero_conect = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_conect)
        attention = F.softmax(attention, dim=1)

        attention = self.dropout(attention)
        adj = self.norm(attention + adj)

        return adj

    # !!! Need optimize !!!
    def _prepare_attentional_mechanism_input(self, x):
        B = x.size()[0]
        N = x.size()[1]

        x_repeated_in_chunks = x.repeat_interleave(N, dim=1)
        x_repeated_alternating = x.repeat(1, N, 1)

        all_combinations_matrix = torch.cat([x_repeated_in_chunks, x_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(B, N * N, 2 * self.d_out)
