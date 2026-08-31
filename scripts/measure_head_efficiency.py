"""Feature-level inference cost of official comparator heads (HLEG / LabCR).

Measures inference-active parameters, per-image FLOPs and batch-1 latency of the
official Query2Label-style heads released by HLEG and LabCR, excluding the shared
frozen backbone. One method per process (both repos ship a top-level ``models``
package, so their imports collide).

Usage:  python scripts/measure_head_efficiency.py {hleg,labcr}
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPOS = {
    "hleg": Path.home() / "lambda/projects/IntentRecognition/HLEG",
    "labcr": Path.home() / "lambda/projects/IntentRecognition/LabCR",
}
# Defaults from each repo's train.py argument parser.
CFG = {
    "hleg": dict(enc_layers=1, dec_layers=3, dim_feedforward=8192, hidden_dim=2048, nheads=4),
    "labcr": dict(enc_layers=1, dec_layers=2, dim_feedforward=8192, hidden_dim=2048, nheads=4),
}
NUM_CLASS = 28
FEATURE_DIM = 1024      # CLIP ViT-L/14 token width
GRID = 7                # HLEG's Linear(49, .) heads fix the token grid to 7x7


class Head(nn.Module):
    """Everything the official model runs after the frozen backbone."""

    def __init__(self, method):
        super().__init__()
        cfg = CFG[method]
        sys.path.insert(0, str(REPOS[method]))
        from models.transformer import build_transformer
        from models.query2label import GroupWiseLinear

        args = argparse.Namespace(
            dropout=0.1, pre_norm=False, dataname="intentonomy",
            keep_other_self_attn_dec=False, keep_first_self_attn_dec=False, **cfg,
        )
        self.method = method
        self.transformer = build_transformer(args)
        d = cfg["hidden_dim"]
        # kept because CLIP token width (1024) != transformer width (2048)
        self.input_proj = nn.Conv2d(FEATURE_DIM, d, kernel_size=1)
        self.query_embed = nn.Embedding(NUM_CLASS, d)
        self.fc = GroupWiseLinear(NUM_CLASS, d, bias=True)
        if method == "hleg":  # extra hierarchical output heads
            self.fc_middle = GroupWiseLinear(15, d, bias=True)
            self.fc_coarse = GroupWiseLinear(9, d, bias=True)

    def forward(self, src, pos):
        hs = self.transformer(self.input_proj(src), self.query_embed.weight, pos)[0]
        if self.method == "hleg":
            return self.fc(hs[0][-1]), self.fc_middle(hs[1][-1]), self.fc_coarse(hs[2][-1])
        return self.fc(hs[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=sorted(CFG))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Head(args.method).to(device).eval()
    src = torch.randn(1, FEATURE_DIM, GRID, GRID, device=device)
    pos = torch.randn(1, CFG[args.method]["hidden_dim"], GRID, GRID, device=device)

    # TransformerDecoder stores an unused prototype layer alongside its clones;
    # it never runs at inference, so it is excluded from the parameter count.
    decoder = model.transformer.decoder
    proto = set(id(p) for p in decoder.decoder_layer.parameters()) \
        if hasattr(decoder, "decoder_layer") else set()
    params = sum(p.numel() for p in model.parameters() if id(p) not in proto)

    from fvcore.nn import FlopCountAnalysis
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (src, pos))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        macs = flops.total()

    try:
        from thop import profile as thop_profile
        import copy
        with torch.no_grad():
            thop_macs, _ = thop_profile(copy.deepcopy(model), inputs=(src, pos), verbose=False)
    except Exception as exc:  # thop is optional
        thop_macs = float("nan")

    with torch.no_grad():
        for _ in range(args.warmup):
            model(src, pos)
        if device == "cuda":
            torch.cuda.synchronize()
        # batch-1 latency: synchronize per iteration, report the median
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(src, pos)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        latency = times[len(times) // 2]

    print(f"method   : {args.method}")
    print(f"device   : {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")
    print(f"params   : {params/1e6:.2f} M ({params})")
    print(f"MACs     : fvcore {macs/1e6:.1f} M -> FLOPs(2x) {2*macs/1e6:.1f} M")
    print(f"MACs     : thop   {thop_macs/1e6:.1f} M -> FLOPs(2x) {2*thop_macs/1e6:.1f} M")
    print(f"latency  : {latency*1e3:.3f} ms")


if __name__ == "__main__":
    main()
