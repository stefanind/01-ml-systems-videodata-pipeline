"""Load cached Qwen weights and caption prepared videos sequentially."""

import time

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from .common import path, read_args, selection


def load_model(config):
    """Load one cached Qwen model and its tokenizer for reuse across videos."""
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])

    # Fail clearly before loading weights if the requested CUDA setup is unavailable.
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; run on a GPU Pod with CUDA-enabled PyTorch"
            )
        torch.cuda.set_device(device)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU needs dtype=float16 in config.json")

    model_dir = path(config, "model_dir")
    print(f"Loading Qwen on {device} with {dtype}...", flush=True)
    load_start = time.perf_counter()

    # Keep the whole model on the configured device. eval() disables training behavior.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(
        f"Model ready on {model.device} with {model.dtype} "
        f"in {time.perf_counter() - load_start:.2f} seconds",
        flush=True,
    )
    return model, tokenizer


@torch.inference_mode()
def caption_video(video, config, model, tokenizer):
    """Generate and save one caption using an already loaded model; return its text."""
    # Inference mode avoids building training gradients during generation.
    video_id = video["video_id"]
    device = model.device
    video_start = time.perf_counter()
    inputs = torch.load(
        path(config, "data_dir", "preprocessed", f"{video_id}.pt"),
        weights_only=True,
    )

    # Preprocessing saved CPU tensors; move those tensors to the model's device now.
    device_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            device_inputs[key] = value.to(device)
        else:
            device_inputs[key] = value
    inputs = device_inputs

    # Greedy decoding picks the highest-scoring next token without sampling.
    generated = model.generate(
        **inputs,
        do_sample=False,
        temperature=1.0,
        max_new_tokens=config["max_new_tokens"],
    )

    # CUDA launches asynchronously. Wait for completion before reporting elapsed time.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # generate() includes the input prompt. Decode only the newly generated tokens.
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_length:].cpu()
    caption = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not caption:
        raise ValueError(f"Qwen returned an empty caption for {video_id}")

    output = path(config, "data_dir", "captions")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{video_id}.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"{video_id}: {caption}", flush=True)
    print(
        f"Finished {video_id} in {time.perf_counter() - video_start:.2f} seconds "
        f"({len(new_tokens)} new tokens)",
        flush=True,
    )
    return caption


def main():
    config, _ = read_args()
    # Loading happens once, outside the per-video loop.
    model, tokenizer = load_model(config)
    videos = selection(config)
    for index, video in enumerate(videos, start=1):
        print(
            f"Captioning {index}/{len(videos)}: {video['video_id']} "
            f"(up to {config['max_new_tokens']} new tokens)...",
            flush=True,
        )
        caption_video(video, config, model, tokenizer)


if __name__ == "__main__":
    main()
