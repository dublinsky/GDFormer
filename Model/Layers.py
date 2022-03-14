import torch.nn as nn

from Model.AGDC.DiffusionAttention import DiffAtten
from Model.AGDC.GraphAttention import GraphAtten


class EncoderLayer(nn.Module):
    def __init__(self, d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes, max_diff_step=3,
                 droprate=0.1, diff_type='Attention', adj_type='D'):
        super(EncoderLayer, self).__init__()

        self.adj_type = adj_type

        if adj_type == 'D':
            self.graphatt = GraphAtten(g_n_head, d_in, nodes, droprate)
        
        self.diffatt = DiffAtten(d_n_head, d_in, d_k, d_v, d_out, nodes, max_diff_step, droprate, diff_type)

    def forward(self, x, adj):
        x = self.diffatt(x, adj, x, adj)
        if self.adj_type == 'D':
            adj = self.graphatt(x, adj)
        return x, adj


class DecoderLayer(nn.Module):
    def __init__(self, d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes, max_diff_step=3,
                 droprate=0.1, diff_type='Attention', adj_type='D'):
        super(DecoderLayer, self).__init__()

        self.adj_type = adj_type

        if adj_type == 'D':
            self.slf_graphatt = GraphAtten(g_n_head, d_in, nodes, droprate)
        
        self.slf_diffatt = DiffAtten(d_n_head, d_in, d_k, d_v, d_out, nodes, max_diff_step, droprate, diff_type)

        # self.enc_graphatt = GraphAtten(g_n_head, d_in, d_out, nodes, droprate)
        self.enc_diffatt = DiffAtten(d_n_head, d_in, d_k, d_v, d_out, nodes, max_diff_step, droprate)

    def forward(self, x, adj, mem_x, mem_adj):
        slf_x = self.slf_diffatt(x, adj, x, adj)
        if self.adj_type == 'D':
            slf_adj = self.slf_graphatt(slf_x, adj)
        else:
            slf_adj = adj
        
        tgt_x = self.enc_diffatt(slf_x, slf_adj, mem_x, mem_adj)
        return tgt_x, slf_adj, mem_x, mem_adj
