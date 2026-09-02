from common_utils import nonzero_averaging
from model.attention_layer import *
from model.sub_layers import *
from model.eligible_pool import EligiblePool
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        # mid_dim = [self.embedding_output_dim] * (self.num_dan_layers - 1)
        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        # Eligible-operation pooling, inserted between DAN layer 0 and layer 1.
        # 'nopooling' keeps the original DAN untouched.
        self.pooling_type = getattr(config, 'pooling_type', 'nopooling')
        self.eligible_pool = None
        if self.pooling_type == 'eligible':
            assert self.num_dan_layers >= 2, \
                "eligible pooling needs at least two DAN layers (pooling sits between layer 0 and 1)"
            self.eligible_pool = EligiblePool()

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                opes_appertain=None, deleted_op_nodes=None, eligible_opes=None):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param opes_appertain: job index per operation [sz_b, N], required for eligible pooling
        :param deleted_op_nodes: bool [sz_b, N], completed/padding nodes, required for eligible pooling
        :param eligible_opes: bool [sz_b, N], protected candidate nodes, required for eligible pooling
        :return:
            fea_j.shape = [sz_b, k, output_dim] (k = N without pooling)
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
            candidate_out: candidate indices valid for fea_j
                           (remapped to pooled positions when eligible pooling is active)
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        candidate_out = candidate

        for layer in range(self.num_dan_layers):
            candidate_idx = candidate_out.unsqueeze(-1). \
                repeat(1, 1, fea_j.shape[-1]).type(torch.int64)

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul,
                                     fea_j_jc).reshape(sz_b, M, M, -1)
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer)

            if self.eligible_pool is not None and layer == 0:
                if opes_appertain is None or deleted_op_nodes is None or eligible_opes is None:
                    raise ValueError(
                        "Eligible pooling is enabled but opes_appertain / deleted_op_nodes / "
                        "eligible_opes were not passed to the model")
                fea_j, op_mask, candidate_out, _ = self.eligible_pool(
                    fea_j, candidate_out, opes_appertain,
                    eligible_opes, deleted_op_nodes)

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global, candidate_out


class DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(
            device)
        self.actor = Actor(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                opes_appertain=None, deleted_op_nodes=None):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """

        # Eligible operations: candidates of jobs that currently have at
        # least one valid (job, machine) pair. Exactly these candidate
        # features enter comp_val via comp_idx, so exactly these nodes are
        # protected from pooling.
        eligible_opes = None
        if opes_appertain is not None:
            N = fea_j.size(1)
            job_eligible = (~dynamic_pair_mask.bool()).any(dim=-1)  # [sz_b, J]
            eligible_opes = torch.zeros(fea_j.size(0), N, dtype=torch.bool,
                                        device=fea_j.device)
            eligible_opes.scatter_(1, candidate.long(), job_eligible)

        fea_j, fea_m, fea_j_global, fea_m_global, candidate_used = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
            opes_appertain=opes_appertain, deleted_op_nodes=deleted_op_nodes,
            eligible_opes=eligible_opes)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate_used.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
