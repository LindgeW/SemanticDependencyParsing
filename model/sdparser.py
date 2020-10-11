import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.rnn import LSTM
from model.sdp_biaff import SDPBiaffine
from modules.layers import Embeddings, CharCNNEmbedding
from modules.dropout import *


class SDParser(nn.Module):
    def __init__(self, num_wds, num_chars, num_tags,
                 wd_embed_dim, char_embed_dim, tag_embed_dim,
                 hidden_size, num_layers,
                 arc_size, rel_size, num_lbl,
                 arc_drop=0.0, rel_drop=0.0, dropout=0.0, embed_weight=None):
        super(SDParser, self).__init__()

        self.dropout = dropout

        self.word_embedding = Embeddings(num_embeddings=num_wds,
                                         embedding_dim=wd_embed_dim,
                                         embed_weight=embed_weight,
                                         pad_idx=0)

        self.char_embedding = CharCNNEmbedding(num_chars, char_embed_dim)

        self.tag_embedding = Embeddings(num_embeddings=num_tags,
                                        embedding_dim=tag_embed_dim,
                                        pad_idx=0)

        self.encoder = LSTM(input_size=wd_embed_dim + char_embed_dim + tag_embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

        self.biaff_scorer = SDPBiaffine(2*hidden_size, arc_size, rel_size, num_lbl, arc_drop, rel_drop)

    def forward(self, wd_ids, char_ids, tag_ids):
        seq_mask = wd_ids.gt(0)
        wd_embed = self.word_embedding(wd_ids)
        char_embed = self.char_embedding(char_ids)
        tag_embed = self.tag_embedding(tag_ids)
        enc_inp = torch.cat((wd_embed, char_embed, tag_embed), dim=-1).contiguous()

        if self.training:
            enc_inp = timestep_dropout(enc_inp, p=self.dropout)

        enc_out = self.encoder(enc_inp, non_pad_mask=seq_mask)[0]

        if self.training:
            enc_out = timestep_dropout(enc_out, p=self.dropout)

        edge_score, lbl_score = self.biaff_scorer(enc_out)

        return edge_score, lbl_score

    def graph_decode(self, edge_score, rel_score, mask):
        '''
        含<root>节点
        :param edge_score: (b, t, t)
        :param rel_score: (b, t, t, c)
        :param mask: (b, t)  1对应有效部分，0为对齐pad
        :return: (b, t, t)
        '''
        lens = mask.sum(dim=1)
        pad_mask = mask.eq(0)
        bz, seq_len, seq_len = edge_score.size()
        # head_tgt: (bz, seq_len, seq_len)
        weights = torch.ones((bz, seq_len, seq_len), dtype=torch.float, device=edge_score.device)
        weights = weights.masked_fill(pad_mask.unsqueeze(1), 0)
        weights = weights.masked_fill(pad_mask.unsqueeze(2), 0)
        head_probs = torch.sigmoid(edge_score).unsqueeze(3)
        label_probs = F.softmax(rel_score, dim=3)
        graph_probs = weights.unsqueeze(3) * head_probs * label_probs  # (b, t, t, c)
        graph_probs = graph_probs.detach().cpu().numpy()

        head_probs = graph_probs.sum(axis=-1)  # (b, t, t)
        head_preds = np.where(head_probs >= 0.5, 1, 0)
        rel_preds = np.argmax(graph_probs, axis=-1)
        masked_head_preds = np.zeros(head_preds.shape, dtype=np.int)
        for i, (hp, length) in enumerate(zip(head_preds, lens)):
            masked_head_preds[i, :length, :length] = hp[:length, :length]

        # refined
        for i, length in enumerate(lens):
            # self circle
            for j in range(length):
                if masked_head_preds[i, j, j] == 1:
                    masked_head_preds[i, j, j] = 0
            # no root
            n_root = np.sum(masked_head_preds[i, :, 0])
            if n_root == 0:
                new_root = np.argmax(head_probs[i, 1:, 0]) + 1
                masked_head_preds[i, new_root, 0] = 1
            elif n_root > 1:  # multi root
                root_ = np.argmax(head_probs[i, 1:, 0]) + 1
                masked_head_preds[i, :, 0] = 0
                masked_head_preds[i, root_, 0] = 1
            # no heads
            n_heads = masked_head_preds[i, :length, :length].sum(axis=-1)
            n_heads[0] = 1
            for j, n_head in enumerate(n_heads):
                if n_head == 0:
                    head_probs[i, j, j] = 0
                    new_head = np.argmax(head_probs[i, j, 1:length]) + 1
                    masked_head_preds[i, j, new_head] = 1
        graph_preds = masked_head_preds + rel_preds * masked_head_preds
        return graph_preds

