"""Load cached Qwen weights and caption prepared videos sequentially."""

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from .common import path, read_args, selection


def main():
    config, _ = read_args()
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run on a GPU Pod with CUDA-enabled PyTorch")
        torch.cuda.set_device(device)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU needs dtype=float16 in config.json")
    model_dir = path(config, "model_dir")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    output = path(config, "data_dir", "captions")
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for video in selection(config):
            video_id = video["video_id"]
            inputs = torch.load(path(config, "data_dir", "preprocessed", f"{video_id}.pt"), weights_only=True)
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=config["max_new_tokens"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            new_tokens = generated[0, inputs["input_ids"].shape[1]:].cpu()
            caption = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not caption:
                raise ValueError(f"Qwen returned an empty caption for {video_id}")
            (output / f"{video_id}.txt").write_text(caption + "\n", encoding="utf-8")
            print(f"{video_id}: {caption}", flush=True)
            del inputs, generated


if __name__ == "__main__":
    main()
