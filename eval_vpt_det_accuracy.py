#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate VPT-Det accuracy with custom Qwen2-VL-VPT classes.

Vision modes:
- --vision-mode dino  (default): DINO-only. Strip "<image>", keep "<detection_image>", and pass detection_images.
- --vision-mode base             : Base-only. Keep "<image>", strip "<detection_image>", pass only base images.
- --vision-mode both             : DINO + Base. Keep both tokens and pass both image paths.

Common behavior:
- Prefill the assistant detection block (<|detection_action_start|> ... <|detection_action_end|>)
  so the model sees the <|detection_action|> token in-context and then only generates the textual answer.
- Left padding, explicit truncation/padding, para_mask disabled for eval.
"""

from __future__ import annotations
import argparse, json, os, re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from PIL import Image

from transformers.models.qwen2_vl_vpt.modeling_qwen2_vl_vpt import (
    VPT_Qwen2VLConfig,
    VPT_Qwen2VLProcessor,
    VPT_Qwen2VLForConditionalGeneration,
)

DA_START = "<|detection_action_start|>"
DA = "<|detection_action|>"
DA_END = "<|detection_action_end|>"

# capture everything AFTER DA_END (gold/pred text)
ANS_AFTER_DA_END = re.compile(rf"{re.escape(DA_END)}\s*(.+)", re.DOTALL)
# capture everything UP TO AND INCLUDING DA_END (prefill block)
PREFILL_UP_TO_DA_END = re.compile(
    rf"{re.escape(DA_START)}.*?{re.escape(DA_END)}", re.DOTALL
)

# Known control/special tokens that can appear in Qwen/Qwen-VL/VPT generations
SPECIAL_TOKEN_PAT = re.compile(
    r"(?:"
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|"
    r"<\|detection_action_start\|>|<\|detection_action_end\|>|<\|detection_action\|>|"
    r"<\|clip_action_start\|>|<\|clip_action_end\|>|<\|clip_action\|>|"
    r"<\|seg_action_start\|>|<\|seg_action_end\|>|<\|seg_action\|>|"
    r"<image>|<detection_image>|<clip_image>|<seg_image>"
    r")",
    flags=re.IGNORECASE,
)

def clean_specials(text: str) -> str:
    """Remove control/special tokens and stray <...> tags, collapse whitespace."""
    text = SPECIAL_TOKEN_PAT.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)  # just in case
    return re.sub(r"\s+", " ", text).strip()

def norm_ans(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

@dataclass
class Sample:
    # full_text: user prompt + assistant prefill block up to DA_END + trailing newline
    full_text: str
    image_path: str
    gold_answer: Optional[str]  # None for test

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def extract_gold_from_messages(messages: List[dict]) -> Optional[str]:
    if not messages or len(messages) < 2:
        return None
    content = messages[1].get("content", "")
    if not content:
        return None
    m = ANS_AFTER_DA_END.search(content)
    if not m:
        return None
    tail = m.group(1)
    tail = clean_specials(tail)
    tail = tail.split("\n", 1)[0].strip()
    return norm_ans(tail) if tail else ""

def extract_prefill_block(messages: List[dict]) -> Optional[str]:
    """
    From assistant content, return the detection prefill block:
    '<|detection_action_start|> ... <|detection_action|>{...} ... <|detection_action_end|>'
    """
    if not messages or len(messages) < 2:
        return None
    content = messages[1].get("content", "")
    if not content:
        return None
    m = PREFILL_UP_TO_DA_END.search(content)
    return m.group(0) if m else None

def build_samples(rows: List[dict]) -> List[Sample]:
    samples: List[Sample] = []
    for r in rows:
        msgs = r.get("messages", [])
        if not msgs or msgs[0].get("role") != "user":
            continue
        user_prompt = msgs[0].get("content", "")
        det_imgs = r.get("detection_images") or []
        if not user_prompt or not det_imgs:
            continue
        prefill = extract_prefill_block(msgs)
        if not prefill:
            # malformed row (no action prefill) -> skip
            continue
        # text presented BEFORE generation:
        # user turn (contains "<image>" and "<detection_image>") + newline + assistant prefill block + newline
        full_text = f"{user_prompt}\n{prefill}\n"
        gold = extract_gold_from_messages(msgs)
        samples.append(Sample(full_text=full_text, image_path=det_imgs[0], gold_answer=gold))
    return samples

def load_model_and_processor(model_dir: str, dtype: torch.dtype, device: str):
    config = VPT_Qwen2VLConfig.from_pretrained(model_dir)
    processor = VPT_Qwen2VLProcessor.from_pretrained(model_dir)
    model = VPT_Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, config=config, torch_dtype=dtype
    ).to(device)
    model.eval()

    tok = processor.tokenizer
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only

    # generation defaults (avoid warnings)
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tok.pad_token_id
    if getattr(model.generation_config, "eos_token_id", None) is None:
        model.generation_config.eos_token_id = tok.eos_token_id

    # Disable paragraph masking for eval
    model.config.para_mask_id = 0

    return model, processor

def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]

def _strip_token(txt: str, token: str) -> str:
    txt = txt.replace(f"{token}\n", "")
    txt = txt.replace(token, "")
    return txt

def prepare_batch(
    processor: VPT_Qwen2VLProcessor,
    batch: List[Sample],
    device: str,
    padding: str = "longest",
    max_text_len: Optional[int] = None,
    vision_mode: str = "dino",  # "dino" | "base" | "both"
):
    """
    - dino : strip '<image>', keep '<detection_image>', pass detection_images only.
    - base : keep '<image>', strip '<detection_image>', pass base images only.
    - both : keep both tokens, pass both base images and detection_images.
    """
    vision_mode = vision_mode.lower()
    assert vision_mode in {"dino", "base", "both"}

    if vision_mode == "dino":
        texts = [_strip_token(s.full_text, "<image>") for s in batch]
        # text only via tokenizer
        enc = processor.tokenizer(
            texts, return_tensors="pt", padding=padding, truncation=True, max_length=max_text_len
        )
    elif vision_mode == "base":
        texts = [_strip_token(s.full_text, "<detection_image>") for s in batch]
        pil_images = [Image.open(s.image_path).convert("RGB") for s in batch]
        enc = processor(
            text=texts, images=pil_images,
            return_tensors="pt", padding=padding, truncation=True, max_length=max_text_len
        )
    else:  # "both"
        texts = [s.full_text for s in batch]
        pil_images = [Image.open(s.image_path).convert("RGB") for s in batch]
        enc = processor(
            text=texts, images=pil_images,
            return_tensors="pt", padding=padding, truncation=True, max_length=max_text_len
        )

    # Move text (and base image) tensors to device
    for k, v in list(enc.items()):
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(device)

    # Attach DINO detection images if needed
    if vision_mode in {"dino", "both"}:
        pil_images_det = [Image.open(s.image_path).convert("RGB") for s in batch]
        det = processor.detection_processor(images=pil_images_det, return_tensors="pt")
        enc["detection_images"] = det["pixel_values"].to(device)

    return enc

def decode_outputs(processor: VPT_Qwen2VLProcessor, token_ids: torch.Tensor) -> List[str]:
    return processor.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

def extract_pred_answer(text: str) -> str:
    m = ANS_AFTER_DA_END.search(text)
    if not m:
        return ""
    tail = m.group(1)
    tail = clean_specials(tail)
    tail = tail.split("\n", 1)[0].strip()
    return norm_ans(tail)

def evaluate(
    model_dir: str,
    data_path: str,
    batch_size: int,
    max_new_tokens: int,
    device: Optional[str] = None,
    dtype_str: str = "bfloat16",
    limit: Optional[int] = None,
    padding: str = "longest",
    vision_mode: str = "dino",   # NEW
    max_text_len: Optional[int] = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_str, torch.bfloat16)

    # reduce OOM risk
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True

    model, processor = load_model_and_processor(model_dir, dtype, device)

    rows = load_jsonl(data_path)
    samples = build_samples(rows)
    if limit:
        samples = samples[:limit]

    n_has_gold = 0
    n_correct = 0
    examined: List[Tuple[str, str, str]] = []

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    with torch.no_grad():
        for batch in batchify(samples, batch_size):
            enc = prepare_batch(
                processor, batch, device,
                padding=padding, max_text_len=max_text_len, vision_mode=vision_mode
            )
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id if pad_id is not None else eos_id,
            )
            texts = decode_outputs(processor, gen_ids)

            for s, t in zip(batch, texts):
                pred = extract_pred_answer(t)
                gold = s.gold_answer
                if gold is not None:
                    n_has_gold += 1
                    if pred == gold:
                        n_correct += 1
                if len(examined) < 25:
                    examined.append((gold if gold is not None else "", pred, s.full_text[:80]))

            del enc, gen_ids
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    acc = (n_correct / n_has_gold) if n_has_gold else 0.0
    print(f"# Samples total: {len(samples)}")
    print(f"# With gold:     {n_has_gold}")
    print(f"# Correct:       {n_correct}")
    print(f"Accuracy:        {acc:.4f}")

    print("\n--- Examples (up to 25) ---")
    for i, (g, p, pr) in enumerate(examined, 1):
        print(f"[{i}] GOLD='{g}' | PRED='{p}' | PROMPT_PREVIEW='{pr}…'")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--padding", type=str, default="longest", choices=["longest", "max_length", "do_not_pad"])
    ap.add_argument("--vision-mode", type=str, default="dino", choices=["dino", "base", "both"],
                    help="Which visual encoder path(s) to use at eval.")
    ap.add_argument("--max-text-len", type=int, default=None, help="Tokenizer max_length when padding=max_length.")
    return ap.parse_args()

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    args = parse_args()
    evaluate(
        model_dir=args.model,
        data_path=args.data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype_str=args.dtype,
        limit=args.limit,
        padding=args.padding,
        vision_mode=args.vision_mode,
        max_text_len=args.max_text_len,
    )
