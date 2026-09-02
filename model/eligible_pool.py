import torch
import torch.nn as nn


class EligiblePool(nn.Module):
    """
    Eligible-operation pooling for the dense DAN tensor format.

    Unlike SAGC, this is not a learned coarsening: it is a hard filter that
    keeps exactly the eligible operations (candidates of jobs that currently
    have at least one valid machine pair) and drops everything else. There
    is no scoring network and no pooling ratio.

    Differences to a plain "keep only eligible" filter, caused by batching:
      * Instances in a batch have different numbers of eligible operations,
        so the pooled tensor width k is the max eligible count in the batch.
        Instances with fewer eligible operations than k get inert filler
        nodes (zeroed embeddings, disconnected in op_mask) in the remaining
        slots so the tensor stays rectangular.
      * DAN has no explicit operation adjacency. The OAB uses a roll trick
        on the dense tensor, so compacting the kept nodes in their original
        order reconnects the chain automatically (o1-o2-o3 with o2 removed
        becomes o1-o3). Only op_mask has to be rebuilt.
    """

    def __init__(self):
        super().__init__()
        self.diagnostic_mode = False
        self._last_diag = None

    @staticmethod
    def _rebuild_op_mask(kept_jobs, kept_filler):
        """
        Rebuild the [B, k, 3] op_mask for the pooled tensor.

        Column 0 masks the predecessor, column 2 the successor, column 1
        (self) is never masked, exactly as in DAN. Position i attends to
        position i-1 only if both belong to the same job and neither is a
        filler node. Same for i+1.
        """
        B, k = kept_jobs.shape
        device = kept_jobs.device
        op_mask = torch.zeros(B, k, 3, dtype=torch.float32, device=device)

        # predecessor side
        op_mask[:, 0, 0] = 1
        if k > 1:
            job_change = (kept_jobs[:, 1:] != kept_jobs[:, :-1])
            cut_pre = job_change | kept_filler[:, 1:] | kept_filler[:, :-1]
            op_mask[:, 1:, 0] = torch.maximum(op_mask[:, 1:, 0], cut_pre.float())

        # successor side
        op_mask[:, -1, 2] = 1
        if k > 1:
            cut_sub = job_change | kept_filler[:, :-1] | kept_filler[:, 1:]
            op_mask[:, :-1, 2] = torch.maximum(op_mask[:, :-1, 2], cut_sub.float())

        return op_mask

    def forward(self, h, candidate, opes_appertain, eligible_opes, deleted_opes):
        """
        :param h:              operation embeddings [B, N, d]
        :param candidate:      candidate operation indices [B, J]
        :param opes_appertain: job index per operation [B, N]
        :param eligible_opes:  bool [B, N], True = must be kept
        :param deleted_opes:   bool [B, N], True = completed or padding
        :return:
            h_pooled          [B, k, d]
            op_mask_pooled    [B, k, 3]
            candidate_pooled  [B, J]  candidate indices remapped to pooled
                              positions (fallback 0 for candidates that are
                              not in the pooled graph, harmless because their
                              comp_idx entries are zero and their actions are
                              masked by dynamic_pair_mask)
            top_idx           [B, k]  pooled position -> original position
        """
        B, N, d = h.shape
        device = h.device

        # 1) keep exactly the eligible, non-deleted operations
        keep_mask = eligible_opes & ~deleted_opes
        sel_scores = keep_mask.float()

        k = max(1, int(keep_mask.sum(dim=-1).max().item()))

        # 2) select and sort, so the compacted tensor keeps the original order
        top_idx = torch.topk(sel_scores, k, dim=-1).indices
        top_idx, _ = torch.sort(top_idx, dim=-1)

        # 3) gather embeddings
        h_pooled = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, d))

        # 4) positions selected as padding (not actually eligible) are inert
        #    filler: zero their embeddings so they behave like DAN's own
        #    deleted nodes (excluded by nonzero_averaging and inert in the
        #    next attention layer)
        kept_filler = ~keep_mask.gather(1, top_idx)
        h_pooled = h_pooled.masked_fill(kept_filler.unsqueeze(-1), 0.0)

        # 5) rebuild op_mask on the pooled tensor
        kept_jobs = opes_appertain.gather(1, top_idx)
        op_mask_pooled = self._rebuild_op_mask(kept_jobs, kept_filler)

        # 6) remap candidate indices to pooled positions
        reverse_map = torch.zeros(B, N, dtype=torch.long, device=device)
        pooled_positions = torch.arange(k, device=device).unsqueeze(0).expand(B, -1)
        reverse_map.scatter_(1, top_idx, pooled_positions)
        candidate_pooled = reverse_map.gather(1, candidate.long())

        if self.diagnostic_mode:
            self._last_diag = {
                "keep_mask": keep_mask.detach().cpu(),
                "top_idx": top_idx.detach().cpu(),
                "k": k,
            }

        return h_pooled, op_mask_pooled, candidate_pooled, top_idx
