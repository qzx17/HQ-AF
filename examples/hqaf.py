#=====================adjust attention_weights=====================
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class hqaf(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "hqaf"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.closs = nn.CrossEntropyLoss(reduction='sum')
        embed_l = d_model

        self.q_embed = nn.Embedding(self.n_pid+1, embed_l)
        self.qa_embed = nn.Embedding(2, embed_l)
        self.c_embed = nn.Embedding(self.n_question + 1, embed_l)
        self.c_embed_difficult = nn.Embedding(52, embed_l)
        self.utT_embedding = nn.Embedding(21, embed_l)
        self.utT_embedding2 = nn.Embedding(21, embed_l)

        self.pt_embedding = nn.Embedding(17, embed_l)
        nn.init.xavier_uniform_(self.c_embed_difficult.weight)
        nn.init.xavier_uniform_(self.pt_embedding.weight)
        nn.init.xavier_uniform_(self.utT_embedding.weight)
        nn.init.xavier_uniform_(self.utT_embedding2.weight)
        nn.init.xavier_uniform_(self.q_embed.weight)
        nn.init.xavier_uniform_(self.qa_embed.weight)
        nn.init.xavier_uniform_(self.c_embed.weight)

        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                            d_model=d_model * 2, d_feature=(d_model * 2) / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

        self.out = nn.Sequential(
            nn.Linear(d_model * 4, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        
        self.simFFN = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.dim() > 0 and p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def compute_differences(self, avg_embeddings, embeddings):
        diff = avg_embeddings.unsqueeze(1) - embeddings.unsqueeze(2)
        return diff
    
    def compute_similarities(self, avg_embeddings, embeddings):
        diff = self.compute_differences(avg_embeddings, embeddings)
        batch_size, seq_len, _, embed_dim = diff.shape
        diff = diff.view(batch_size * seq_len * seq_len, embed_dim)
        similarities = self.simFFN(diff)
        similarities = similarities.view(batch_size, seq_len, seq_len)
        return similarities
    
    # MODIFIED: added return_attn argument
    def forward(self, q_data, target, pid_data=None, qtest=False, cd=None, qd=None, qu=None, cutT = None, cpT = None, sdshft = None, sm = None, return_attn=False):
        
        q_embed_data = self.q_embed(pid_data) + self.c_embed(q_data)
        qa_embed_data = q_embed_data + self.qa_embed(target)
    
        utT_embeddings = self.utT_embedding(cutT)
        pt_embeddings = self.pt_embedding(cpT)
        avg_utT_embeddings = self.utT_embedding2(qu)

        utT_similarities = self.compute_similarities(avg_utT_embeddings, utT_embeddings)
        
        c_diff = self.c_embed_difficult(cd)
        c_diff_add_pt_embeddings = c_diff + pt_embeddings
        q_embed_data = torch.cat([q_embed_data, c_diff_add_pt_embeddings], dim=-1)
        qa_embed_data = torch.cat([qa_embed_data, c_diff_add_pt_embeddings], dim=-1)

        # MODIFIED: capture attention weights
        d_output, attn_weights = self.model(q_embed_data, qa_embed_data, utT_similarities, c_diff, return_attn=return_attn)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        if return_attn:
            return preds, attn_weights

        if not qtest:
            return preds, "a"
        else:
            return preds, concat_q


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.batch_counter = 0

        if model_type in {'hqaf'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads, d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads, d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks*2)
            ])

        # Question Correction Module (QC/SC layers omitted for brevity, same as original)
        self.qc1 = nn.Sequential(nn.Linear(d_model, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, d_model), nn.Tanh())
        self.qc2 = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, d_model), nn.Tanh())
        self.qc3 = nn.Sequential(nn.Linear(d_model, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, d_model), nn.Tanh())
        
        self.sc1 = nn.Sequential(nn.Linear(d_model,d_model), nn.Sigmoid())
        self.sc2 = nn.Sequential(nn.Linear(d_model,d_model), nn.Sigmoid())
        self.sc3 = nn.Sequential(nn.Linear(d_model,d_model), nn.Sigmoid())

    def forward(self, q_embed_data, qa_embed_data, utT_similarities, c_diff, return_attn=False):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1: 
            y, _ = block(mask=1, query=y, key=y, values=y, utT_similarities=utT_similarities) 

        init_x = x
        
        # We capture the attention weights from the DC module blocks (blocks_2)
        collected_attentions = [] 

        flag_first = True
        for i, block in enumerate(self.blocks_2):
            if flag_first:  # peek current question (Self Attention)
                x, attn = block(mask=1, query=x, key=x, values=x, utT_similarities=utT_similarities, apply_pos=False)
                if return_attn:
                    collected_attentions.append(attn) # Store attention weights
                flag_first = False
            else:  # dont peek current response
                x, _ = block(mask=0, query=x, key=x, values=y, utT_similarities=utT_similarities, apply_pos=True)
                flag_first = True
                
                if i == 1:
                    SDF1 = self.qc1(x - init_x)  
                    SC1 = self.sc1(x - init_x) 
                    x = SC1 * x + SDF1 * (1 - SC1)
                elif i == 3:
                    SDF2 = self.qc2(c_diff)  
                    SC2 = self.sc2(x - init_x)  
                    x = SC2 * x + SDF2 * (1 - SC2)
                elif i == 5:
                    SDF3 = self.qc3(x - init_x) 
                    SC3 = self.sc3(x - init_x) 
                    x = SC3 * x + SDF3 * (1 - SC3)
                    
        return x, collected_attentions


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout,  kq_same, emb_type):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, utT_similarities=None, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        
        query2, attn_score = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=(mask==0), utT_similarities=utT_similarities)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query, attn_score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

    def forward(self, q, k, v, mask, zero_pad, utT_similarities=None):
        bs = q.size(0)
        # Assuming emb_type starts with "qid" as per standard usage in this context
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        gammas = self.gammas
        scores, output_attn = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, utT_similarities, gammas)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output, output_attn # Return actual attention scores

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, utT_similarities, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    utT_similarities_expanded = utT_similarities.unsqueeze(1).expand(-1, head, -1, -1)
    scores = scores * utT_similarities_expanded

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp((disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
        
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    total_effect = torch.clamp(torch.clamp((dist_scores*gamma).exp(), min=1e-5), max=1e5) 
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1) # (BS, Heads, SeqLen, SeqLen)

    # Clone scores to return for visualization/analysis
    attn_weights = scores.clone()

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn_weights

