import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


class DiffAtten(nn.Module):
    def __init__(self, n_head, d_in, d_k, d_v, d_out, nodes, max_diff_step=3, droprate=0.1, diff_type='Attention'):
        super(DiffAtten, self).__init__()
        self.n_head = n_head
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.nodes = nodes
        self.max_diff_step=max_diff_step
        self.droprate = droprate
        self.diff_type = diff_type

        if diff_type == 'Attention':
            self.multiatt = MultiHeadAttention(n_head, d_in, d_k, d_v, droprate)
        elif diff_type == 'PPR':
            alpha = torch.tensor(0.)
            self.alpha = nn.parameter.Parameter(alpha)
        elif diff_type == 'HK':
            t = torch.tensor(1.)
            self.t = nn.parameter.Parameter(t)

        self.conv = Linear(d_in, d_out)

        self.dropout = nn.Dropout(droprate)
        self.norm = nn.LayerNorm(d_in)
        self.activation = nn.ReLU()
        self.linear = Linear(d_out, d_in)

    def forward(self, q_x, q_adj, v_x, v_adj):
        sz_b = q_x.size()[0]
        residual = q_x
        
        if self.diff_type == 'Attention':
            q_x1, v_x1 = torch.bmm(q_adj, q_x), torch.bmm(v_adj, v_x)

            q_x = torch.cat([q_x, q_x1], dim=1)
            v_x = torch.cat([v_x, v_x1], dim=1)

            for _ in range(2, self.max_diff_step):
                q_x2, v_x2 = torch.bmm(q_adj, q_x1), torch.bmm(v_adj, v_x1)
                q_x = torch.cat([q_x, q_x2], dim=1)
                v_x = torch.cat([v_x, v_x2], dim=1)
                q_x1, v_x1 = q_x2, v_x2
            
            # q_x = torch.reshape(q_x, [sz_b, self.nodes * self.max_diff_step, -1])
            # v_x = torch.reshape(v_x, [sz_b, self.nodes * self.max_diff_step, -1])
            q_x = F.layer_norm(q_x, [q_x.size()[-1]])
            v_x = F.layer_norm(v_x, [v_x.size()[-1]])
            x = self.multiatt(q_x, q_x, v_x)

            x = torch.reshape(x, [sz_b, self.nodes, -1, self.max_diff_step])
            x = torch.mean(x, dim=-1)
        elif self.diff_type == 'PPR':
            alpha = 1. / (1. + torch.exp(self.alpha))
            alpha_ = 1 - alpha

            q_x1 = self.alpha * alpha_ * torch.bmm(q_adj, q_x)
            q_x = torch.cat([q_x, q_x1], dim=1)

            for _ in range(2, self.max_diff_step):
                q_x2 = alpha_ * torch.bmm(q_adj, q_x1)
                q_x = torch.cat([q_x, q_x2], dim=1)
                q_x1 = q_x2
            
            x = torch.mean(torch.reshape(q_x, [sz_b, self.nodes, -1, self.max_diff_step]), dim=-1)
        elif self.diff_type == 'HK':
            q_x1 = torch.exp(-self.t) * self.t * torch.bmm(q_adj, q_x)
            q_x = torch.cat([q_x, q_x1], dim=1)

            for k in range(2, self.max_diff_step):
                q_x2 = self.t * torch.bmm(q_adj, q_x1) / k
                q_x = torch.cat([q_x, q_x2], dim=1)
                q_x1 = q_x2
            x = torch.mean(torch.reshape(q_x, [sz_b, self.nodes, -1, self.max_diff_step]), dim=-1)
        
        x1 = self.dropout(self.activation(self.conv(x)))

        x = residual + self.linear(x1)
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_in, d_k, d_v, droprate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_in, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_in, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_in, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_in, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(droprate)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, _ = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
