import torch.nn as nn

from Model.Layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes, n_layers=3,
                 max_diff_step=3, droprate=0.1, diff_type='Attention', adj_type='D'):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes,
                         max_diff_step, droprate, diff_type, adj_type)
            for _ in range(n_layers)
        ])

        # self.norm_x = nn.LayerNorm(d_in)
        # self.norm_adj = nn.LayerNorm([nodes, nodes])

    def forward(self, x, adj):
        enc_x, enc_adj = x, adj

        for idx, enc_layer in enumerate(self.layers):
            # print(f'{idx} encoder layers')
            enc_x, enc_adj = enc_layer(enc_x, enc_adj)

        # enc_x, enc_adj = self.norm_x(enc_x), self.norm_adj(enc_adj)
        
        return enc_x, enc_adj


class Decoder(nn.Module):
    def __init__(self, d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes, n_layers=3,
                 max_diff_step=3, droprate=0.1, diff_type='Attention', adj_type='D'):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes,
                         max_diff_step, droprate, diff_type, adj_type)
            for _ in range(n_layers)
        ])

        # self.norm_x = nn.LayerNorm(d_in)
        # self.norm_adj = nn.LayerNorm([nodes, nodes])
    
    def forward(self, x, adj, enc_x, enc_adj):
        dec_x, dec_adj = x, adj

        for idx, dec_layer in enumerate(self.layers):
            # print(f'{idx} decoder layer')
            dec_x, dec_adj, enc_x, enc_adj = dec_layer(dec_x, dec_adj, enc_x, enc_adj)
        
        # dec_x = self.norm_x(dec_x)
        return dec_x, dec_adj


class GDF(nn.Module):
    def __init__(self, d_src, d_tgt, d_in, g_n_head, d_n_head, d_k, d_v, d_out, d_prd,
                 nodes, enc_layers=3, dec_layers=3, max_diff_step=3, droprate=0.1,
                 diff_type='Attention', adj_type='D', dynamic_adj_saved=False):
        super(GDF, self).__init__()

        if adj_type == 'S':
            dynamic_adj_saved = False
        
        self.diff_type = diff_type
        self.adj_type = adj_type
        self.adj_saved = dynamic_adj_saved

        if dynamic_adj_saved:
            self.enc_adj = None
            self.dec_adj = None

        self.srcemb = nn.Linear(d_src, d_in, bias=False)
        self.tgtemb = nn.Linear(d_tgt, d_in, bias=False)

        self.encoder = Encoder(d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes, enc_layers,
                               max_diff_step, droprate, diff_type, adj_type)
        self.decoder = Decoder(d_in, g_n_head, d_n_head, d_k, d_v, d_out, nodes, dec_layers,
                               max_diff_step, droprate, diff_type, adj_type)

        self.fc = nn.Linear(d_in, d_prd, bias=False)

    def forward(self, src_x, src_adj, tgt_x, tgt_adj):
        src_x = self.srcemb(src_x)
        tgt_x = self.tgtemb(tgt_x)

        enc_x, enc_adj = self.encoder(src_x, src_adj)
        dec_x, dec_adj = self.decoder(tgt_x, tgt_adj, enc_x, enc_adj)
        res = self.fc(dec_x)

        if self.adj_saved:
            self.enc_adj = enc_adj
            self.dec_adj = dec_adj
        
        return res
