from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info
import argparse
import json
import os
import re
from string import Template
from datetime import datetime

import cv2
import numpy as np
import torch


VS2_QUERY_TEMPLATE = Template("""
You are an expert for evaluating AI-generated videos from three dimensions:
(1) visual quality - clarity, smoothness, artifacts;
(2) text-to-video alignment - fidelity to the prompt;
(3) physical/common-sense consistency - naturalness and physics plausibility.

Video prompt: $t2v_prompt

Please output in this format:
visual quality: <v_score>;
text-to-video alignment: <t_score>,
physical/common-sense consistency: <p_score>
""")

SCORE_PATTERN = re.compile(
    r"visual quality[^:]*:\s*(\d+).*?"
    r"text-to-video alignment[^:]*:\s*(\d+).*?"
    r"physical/common-sense consistency[^:]*:\s*(\d+)",
    re.DOTALL | re.IGNORECASE,
)

DIMENSIONS = [
    ("v", "visual quality:"),
    ("t", "text-to-video alignment:"),
    ("p", "physical/common-sense consistency:"),
]


def _get_video_fps(url_or_p: str):
    cap = cv2.VideoCapture(url_or_p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url_or_p}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def _find_score_token_index_by_prompt(prompt_text, tokenizer, gen_ids):
    gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)
    key = prompt_text.rstrip(":")
    pattern = r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(key) + r"[^:]*:"
    match = re.search(pattern, gen_str, flags=re.IGNORECASE)
    if not match:
        return -1

    after_text = gen_str[match.end():]
    num_match = re.search(r"\d", after_text)
    if not num_match:
        return -1

    target_substr = gen_str[:match.end() + num_match.start() + 1]
    for i in range(len(gen_ids)):
        partial = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=False)
        if partial == target_substr:
            return i
    return -1


def _get_score_distribution(token_idx, scores, tokenizer):
    if token_idx < 0:
        return []

    logits = scores[token_idx][0]
    log_probs = torch.log_softmax(logits, dim=-1)
    score_probs = []
    for score in range(1, 6):
        ids = tokenizer.encode(str(score), add_special_tokens=False)
        if len(ids) != 1:
            continue
        prob = float(np.exp(log_probs[ids[0]].item()))
        score_probs.append((score, prob))
    return score_probs


def _compute_dimension_metrics(hard_score, score_probs):
    if hard_score is None or not score_probs:
        return {
            "expected_score": None,
            "hard_score": hard_score,
            "confidence": None,
        }

    _, probs = zip(*score_probs)
    total_prob = sum(probs)
    if total_prob <= 0:
        return {
            "expected_score": None,
            "hard_score": hard_score,
            "confidence": None,
        }

    normalized = [(score, prob / total_prob) for score, prob in score_probs]
    expected_score = round(sum(score * prob for score, prob in normalized), 4)
    confidence = round(max(prob for _, prob in normalized), 4)
    return {
        "expected_score": expected_score,
        "hard_score": hard_score,
        "confidence": confidence,
    }


def _resolve_video_paths(video_paths, video_dir):
    resolved = []
    if video_paths:
        for path in video_paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                raise ValueError(f"Video not found: {path}")
            resolved.append(abs_path)

    if video_dir:
        abs_dir = os.path.abspath(video_dir)
        if not os.path.isdir(abs_dir):
            raise ValueError(f"Video directory not found: {video_dir}")
        for name in sorted(os.listdir(abs_dir)):
            if os.path.splitext(name)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                resolved.append(os.path.join(abs_dir, name))

    deduped = []
    seen = set()
    for path in resolved:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    if not deduped:
        raise ValueError("No videos provided. Use --video_path or --video_dir.")
    return deduped


class VideoScore2BatchInferencer:
    def __init__(self, model_name):
        print(f"[Init] Loading model: {model_name}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    def evaluate_video(self, video_path, t2v_prompt, infer_fps=2.0, max_tokens=1024, temperature=0.7):
        if not os.path.exists(video_path):
            raise ValueError(f"Video not found: {video_path}")

        if infer_fps == "raw":
            infer_fps = _get_video_fps(video_path)

        user_prompt = VS2_QUERY_TEMPLATE.substitute(t2v_prompt=t2v_prompt)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": infer_fps},
                {"type": "text", "text": user_prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=infer_fps,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=temperature,
        )

        sequences = gen_out.sequences
        scores = gen_out.scores
        input_len = inputs["input_ids"].shape[1]
        gen_token_ids = sequences[0, input_len:].tolist()
        output_text = self.processor.batch_decode(
            sequences[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        match = SCORE_PATTERN.search(output_text)
        hard_scores = {
            "v": int(match.group(1)) if match else None,
            "t": int(match.group(2)) if match else None,
            "p": int(match.group(3)) if match else None,
        }

        result = {
            "video_name": os.path.splitext(os.path.basename(video_path))[0],
        }
        for dim_key, prompt_text in DIMENSIONS:
            token_idx = _find_score_token_index_by_prompt(prompt_text, self.tokenizer, gen_token_ids)
            score_probs = _get_score_distribution(token_idx, scores, self.tokenizer)
            metrics = _compute_dimension_metrics(hard_scores[dim_key], score_probs)
            result[f"{dim_key}_expected_score"] = metrics["expected_score"]
            result[f"{dim_key}_hard_score"] = metrics["hard_score"]
            result[f"{dim_key}_confidence"] = metrics["confidence"]

        return result, output_text


def main(args):
    video_paths = _resolve_video_paths(args.video_path, args.video_dir)
    inferencer = VideoScore2BatchInferencer(args.model_name)

    batch_result = {
        "prompt": args.t2v_prompt,
        "num_videos": len(video_paths),
        "videos": [],
    }

    for video_path in video_paths:
        result, output_text = inferencer.evaluate_video(
            video_path=video_path,
            t2v_prompt=args.t2v_prompt,
            infer_fps=args.infer_fps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        batch_result["videos"].append(result)

        print("\n[Raw Model Output]")
        print(output_text)
        print("\n====== Inference Result ======")
        print(f"Video Path: {video_path}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("==============================\n")

    final_json = json.dumps(batch_result, indent=2, ensure_ascii=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    if args.save_path:
        save_name = os.path.basename(args.save_path)
        stem, ext = os.path.splitext(save_name)
        if not ext:
            ext = ".json"
        save_path = os.path.abspath(os.path.join("outputs", f"{stem}_{timestamp}{ext}"))
    else:
        save_path = os.path.abspath(os.path.join("outputs", f"vs2_inference_{timestamp}.json"))

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(final_json)
    print(f"Saved results to: {save_path}")

    print(final_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoScore2 inference with structured 3x3 scores")
    parser.add_argument("--video_path", type=str, nargs="*", default=None, help="One or more input video paths")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory of input videos")
    parser.add_argument("--t2v_prompt", type=str, required=True, help="Shared text prompt for the input videos")
    parser.add_argument("--model_name", type=str, default="/mnt/data/wangqq/models/VideoScore2", help="Model name or path")
    parser.add_argument("--infer_fps", default=2.0, help="Inference fps or 'raw'")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--save_path", type=str, default=None, help="Optional output file name under outputs/; a timestamp is always appended to avoid overwriting")
    args = parser.parse_args()

    main(args)
