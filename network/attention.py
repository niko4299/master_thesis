import math

import torch
import torch.nn.functional as F
from torch import nn

from network import clones


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1) 
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1) 
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, asset_atten, h, d_model, dropout, local_context_length, device):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.local_context_length = local_context_length
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.conv_q = nn.Conv2d(d_model, d_model, (1, 1), stride=1, padding=0, bias=True)
        self.conv_k = nn.Conv2d(d_model, d_model, (1, 1), stride=1, padding=0, bias=True)

        self.ass_linears_v = nn.Linear(d_model, d_model)
        self.ass_conv_q = nn.Conv2d(d_model, d_model, (1, 1), stride=1, padding=0, bias=True)
        self.ass_conv_k = nn.Conv2d(d_model, d_model, (1, 1), stride=1, padding=0, bias=True)

        self.attn = None
        self.attn_asset = None
        self.dropout = nn.Dropout(p=dropout)
        self.feature_weight_linear = nn.Linear(d_model, d_model)
        self.asset_atten = asset_atten
        self.device = device

    def forward(self, query, key, value, mask, padding_price_q, padding_price_k):
        if mask is not None:
            mask = mask.unsqueeze(1)  
            mask = mask.repeat(query.size()[0], 1, 1, 1) 
        q_size0 = query.size(0)  
        q_size1 = query.size(1)  
        q_size2 = query.size(2)  
        q_size3 = query.size(3)  
        key_size0 = key.size(0)
        key_size1 = key.size(1)
        key_size2 = key.size(2)
        key_size3 = key.size(3)
        
        if padding_price_q is not None:
            padding_price_q = padding_price_q.permute((1, 3, 0, 2))  
            padding_q = padding_price_q
        else:
            if self.local_context_length > 1:
                padding_q = torch.zeros((q_size1, q_size3, q_size0, self.local_context_length - 1)).to(self.device)
            else:
                padding_q = None
        query = query.permute((1, 3, 0, 2))
        if padding_q is not None:
            query = torch.cat([padding_q, query], -1)

        query = self.conv_q(query)
        query = query.permute((0, 2, 3, 1))
        local_weight_q = torch.matmul(query[:, :, self.local_context_length - 1:, :],
                                      query.transpose(-2, -1)) / math.sqrt(q_size3)  
        
        local_weight_q_list = [F.softmax(local_weight_q[:, :, i: i + 1, i: i + self.local_context_length], dim=-1) for i
                               in range(q_size2)]
        local_weight_q_list = torch.cat(local_weight_q_list, 3)

        local_weight_q_list = local_weight_q_list.permute(0, 1, 3, 2)
      
        q_list = [query[:, :, i: i + self.local_context_length, :] for i in range(q_size2)]
        q_list = torch.cat(q_list, 2)

        query = local_weight_q_list * q_list
        query = query.contiguous().view(q_size1, q_size0, self.local_context_length, q_size2, q_size3)
        query = torch.sum(query, 2)
        query = query.permute((0, 3, 1, 2))
        
        query = query.permute((2, 0, 3, 1)) 
        query = query.contiguous().view(q_size0 * q_size1, q_size2, q_size3)
        query = query.contiguous().view(q_size0 * q_size1, q_size2, self.h, self.d_k).transpose(1,2)  
        if padding_price_k is not None:
            padding_price_k = padding_price_k.permute((1, 3, 0, 2)) 
            padding_k = padding_price_k
        else:
            if self.local_context_length > 1:
                padding_k = torch.zeros((key_size1, key_size3, key_size0, self.local_context_length - 1)).to(self.device)
            else:
                padding_k = None
        key = key.permute((1, 3, 0, 2))
        if padding_k is not None:
            key = torch.cat([padding_k, key], -1)

        key = self.conv_k(key)
        key = key.permute((0, 2, 3, 1))  
        local_weight_k = torch.matmul(key[:, :, self.local_context_length - 1:, :], key.transpose(-2, -1)) / math.sqrt(key_size3)  
        local_weight_k_list = [F.softmax(local_weight_k[:, :, i:i + 1, i:i + self.local_context_length], dim=-1) for i
                               in range(key_size2)]
        local_weight_k_list = torch.cat(local_weight_k_list, 3)

        local_weight_k_list = local_weight_k_list.permute(0, 1, 3, 2)
        k_list = [key[:, :, i:i + self.local_context_length, :] for i in range(key_size2)]
        k_list = torch.cat(k_list, 2)

        key = local_weight_k_list * k_list

        key = key.contiguous().view(key_size1, key_size0, self.local_context_length, key_size2, key_size3)

        key = torch.sum(key, 2)
        key = key.permute((0, 3, 1, 2))
        key = key.permute((2, 0, 3, 1))
        key = key.contiguous().view(key_size0 * key_size1, key_size2, key_size3)
        key = key.contiguous().view(key_size0 * key_size1, key_size2, self.h, self.d_k).transpose(1, 2)
        value = value.view(key_size0 * key_size1, key_size2, key_size3)
        nbatches = q_size0 * q_size1
        value = self.linears[0](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  
        x, self.attn = attention(query, key, value, mask=None,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = x.view(q_size0, q_size1, q_size2, q_size3)
        if (self.asset_atten):
            ass_query = x.permute((2, 1, 0, 3))
            ass_query = ass_query.contiguous().view(q_size2 * q_size1, q_size0, q_size3) 
            ass_query = ass_query.contiguous().view(q_size2 * q_size1, q_size0, self.h, self.d_k).transpose(1, 2)  
            ass_key = x.permute((2, 1, 0, 3)) 
            ass_key = ass_key.contiguous().view(q_size2 * q_size1, q_size0,
                                                q_size3)  
            ass_key = ass_key.contiguous().view(q_size2 * q_size1, q_size0, self.h, self.d_k).transpose(1, 2) 
            ass_value = x.permute((2, 1, 0, 3))
            ass_value = ass_value.contiguous().view(q_size2 * q_size1, q_size0,q_size3)  
            ass_value = ass_value.contiguous().view(q_size2 * q_size1, -1, self.h, self.d_k).transpose(1, 2)
            x, self.attn_asset = attention(ass_query, ass_key, ass_value, mask=None,
                                           dropout=self.dropout)
            x = x.transpose(1, 2).contiguous().view(q_size2 * q_size1, -1,
                                                    self.h * self.d_k)  
            x = x.view(q_size2, q_size1, q_size0, q_size3) 
            x = x.permute(2, 1, 0, 3)
        return self.linears[-1](x)
