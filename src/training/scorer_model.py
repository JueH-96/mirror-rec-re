"""Trainable LLM reranker: backbone + per-candidate score head (Study 3).

One forward per (request, candidate set): the prompt lists candidates with
[ITEM] end-markers; the score head reads the hidden state at each marker.

Score composition (config-controlled):
  base:    s = head(h_i)                          (S2..S5 baselines)
  ordinal: s = head(h_i) with level embedding     (S6: representation only)
  mirror:  s = b(h_i) + g_a(l) * h_util(h_i) + r  (S8: §6.2 monotone structure)
g_a is a per-attribute softplus-cumsum over level increments -> nondecreasing
in strength level by construction (§6.2).
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTRS = ["price", "weight", "battery_life", "storage"]


class RerankScorer(nn.Module):
    def __init__(self, model_name, mode="base", lora_r=16, torch_dtype=torch.bfloat16,
                 g_const=False, g_shared=False):
        super().__init__()
        self.mode = mode
        # study7 ablation switches (mirror mode only, default off):
        # g_const: g(l) frozen to g0 (no strength mapping); g_shared: one
        # shared g row for all attributes (no attribute conditioning)
        self.g_const = g_const
        self.g_shared = g_shared
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation="sdpa")
        backbone.gradient_checkpointing_enable()
        lcfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        self.backbone = get_peft_model(backbone, lcfg)
        hid = backbone.config.hidden_size
        self.head_base = nn.Sequential(nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, 1))
        if mode == "ordinal":
            self.level_emb = nn.Embedding(5, hid)  # levels 1..4 (0 unused)
        if mode == "mirror":
            self.head_util = nn.Sequential(nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, 1))
            self.head_resid = nn.Sequential(nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, 1))
            # per-attribute level increments; g(l) = g0 + sum_{t<=l} softplus(eta_t)
            self.g0 = nn.Parameter(torch.zeros(len(ATTRS)))
            self.eta = nn.Parameter(torch.zeros(len(ATTRS), 4))
        for m in [self.head_base] + ([self.head_util, self.head_resid] if mode == "mirror" else []):
            m.to(torch_dtype)
        if mode == "ordinal":
            self.level_emb.to(torch_dtype)

    def g_value(self, attr_idx, level):
        """Monotone strength mapping per §6.2 (levels 1..4)."""
        if self.g_shared:
            attr_idx = 0  # single shared row: no attribute conditioning
        if self.g_const:
            return self.g0[attr_idx] + 0.0 * self.eta[attr_idx].sum()
        inc = torch.nn.functional.softplus(self.eta[attr_idx])
        return self.g0[attr_idx] + inc[:level].sum()

    def forward(self, input_ids, attention_mask, marker_positions, attr_idx, level):
        """marker_positions: (n_items,) token index of each candidate's marker.
        Returns (n_items,) scores."""
        out = self.backbone(input_ids=input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            output_hidden_states=True)
        h = out.hidden_states[-1][0]  # (seq, hid)
        hi = h[marker_positions]      # (n_items, hid)
        if self.mode == "ordinal":
            hi = hi + self.level_emb(torch.full((hi.size(0),), level, device=hi.device,
                                                dtype=torch.long))
        if self.mode == "mirror":
            b = self.head_base(hi).squeeze(-1)
            u = self.head_util(hi).squeeze(-1)
            r = 0.1 * self.head_resid(hi).squeeze(-1)
            g = self.g_value(attr_idx, level).to(b.dtype)
            return b + g * u + r
        return self.head_base(hi).squeeze(-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
