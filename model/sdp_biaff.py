import torch.nn as nn
from modules.layers import NonlinearMLP, Bilinear, Biaffine
from modules.dropout import *


class SDPBiaffine(nn.Module):
    def __init__(self, input_size,
                 edge_size,
                 label_size,
                 num_lbl,
                 edge_drop=0.33,
                 lbl_drop=0.33):
        super(SDPBiaffine, self).__init__()

        self.edge_size = edge_size
        self.label_size = label_size
        self.edge_drop = edge_drop
        self.lbl_drop = lbl_drop

        self._act = nn.ELU()
        # self.mlp_edge = NonlinearMLP(in_feature=input_size, out_feature=edge_size * 2, activation=self._act)
        # self.mlp_label = NonlinearMLP(in_feature=input_size, out_feature=label_size * 2, activation=self._act)

        self.mlp_head = NonlinearMLP(in_feature=input_size, out_feature=edge_size + label_size, activation=self._act)
        self.mlp_dep = NonlinearMLP(in_feature=input_size, out_feature=edge_size + label_size, activation=self._act)

        # self.edge_biaff = Bilinear(edge_size, edge_size, 1, use_input_bias=True)   # biaffine
        # self.label_biaff = Bilinear(label_size, label_size, num_lbl, use_input_bias=True)  # bilinear
        self.edge_biaff = Biaffine(edge_size, 1,  bias=(True, True))
        self.label_biaff = Biaffine(label_size, num_lbl, bias=(True, True))

    def forward(self, enc_hn):

        # edge_feat = self.mlp_edge(enc_hn)
        # lbl_feat = self.mlp_label(enc_hn)
        # edge_head, edge_dep = edge_feat.chunk(2, dim=-1)
        # lbl_head, lbl_dep = lbl_feat.chunk(2, dim=-1)

        head_feat = self.mlp_head(enc_hn)
        dep_feat = self.mlp_dep(enc_hn)
        edge_head, lbl_head = head_feat.split(self.edge_size, dim=-1)
        edge_dep, lbl_dep = dep_feat.split(self.edge_size, dim=-1)

        if self.training:
            edge_head = timestep_dropout(edge_head, self.edge_drop)
            edge_dep = timestep_dropout(edge_dep, self.edge_drop)

        # (bz, len, len)
        edge_score = self.edge_biaff(edge_dep, edge_head)

        if self.training:
            lbl_head = timestep_dropout(lbl_head, self.lbl_drop)
            lbl_dep = timestep_dropout(lbl_dep, self.lbl_drop)

        # (bz, len, len, num_lbl)
        lbl_score = self.label_biaff(lbl_dep, lbl_head)

        return edge_score, lbl_score
