"""Extract trainable-only weights (LoRA + heads + monotone params) from a fat
checkpoint, writing <run_dir>/final_trainable.pt. Usage: extract_trainable.py <run_dir>"""
import os
import sys

import torch

run_dir = sys.argv[1]
src = os.path.join(run_dir, "final.pt")
state = torch.load(src, map_location="cpu", weights_only=False)


def keep(k):
    return ("lora_" in k) or k.startswith(("head_base", "head_util", "head_resid",
                                           "level_emb", "g0", "eta"))


small = {"model": {k: v for k, v in state["model"].items() if keep(k)},
         "opt": state["opt"], "step": state["step"]}
dst = os.path.join(run_dir, "final_trainable.pt")
torch.save(small, dst)
print("wrote", dst, round(os.path.getsize(dst) / 1e6, 1), "MB;",
      len(small["model"]), "tensors kept")
