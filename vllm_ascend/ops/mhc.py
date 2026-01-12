import torch
import torch.nn.functional as F

def hc_split_sinkhorn_ref(
    mixes_bs: torch.Tensor,       # [bs, (2 + hc) * hc]
    hc_scale: torch.Tensor,       # [3]
    hc_base: torch.Tensor,        # [(2 + hc) * hc]
    hc_mult: int = 4,             # hc
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    # pre: [bs, hc]
    mixes_pre = mixes_bs[:, :hc_mult]
    pre = torch.sigmoid(hc_scale[0] * mixes_pre + hc_base[:hc_mult])

    # post: [bs, hc]
    mixes_post = mixes_bs[:, hc_mult:2 * hc_mult]
    post = 2.0 * torch.sigmoid(hc_scale[1] * mixes_post + hc_base[hc_mult:2 * hc_mult])

    # comb: [bs, hc, hc]
    mixes_comb = mixes_bs[:, 2 * hc_mult:]
    comb = (hc_scale[2] * mixes_comb + hc_base[2 * hc_mult:]).reshape(-1, hc_mult, hc_mult)
    comb = F.softmax(comb, dim=-1) + eps

    for _ in range(sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb