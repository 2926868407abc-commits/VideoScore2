from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info
import os
import re

import cv2
import numpy as np
import torch


def _get_video_fps(url_or_p: str):
    cap = cv2.VideoCapture(url_or_p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url_or_p}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


class eval_VideoScore2_float:
    def __init__(self, model_name: str, floating_method: str = "expected"):
        self.model, self.processor = self.load_model_processor(model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
            )

        alias_map = {
            "normed": "confidence_adjusted",
            "weighted": "expected",
            "expected": "expected",
            "confidence_adjusted": "confidence_adjusted",
        }
        self.floating_method = alias_map.get(floating_method.lower())
        assert self.floating_method in ["expected", "confidence_adjusted"], (
            f"invalid floating_method: {floating_method}"
        )

    def load_model_processor(self, model_name):
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor

    def evaluate_video(self, user_prompt: str, video_path: str, kwargs: dict) -> str | None:
        metrics, output_text = self.evaluate_video_structured(user_prompt, video_path, kwargs)
        if self.floating_method == "expected":
            return (
                metrics["v_expected_score"],
                metrics["t_expected_score"],
                metrics["p_expected_score"],
                output_text,
            )

        return (
            metrics["v_confidence_adjusted_score"],
            metrics["t_confidence_adjusted_score"],
            metrics["p_confidence_adjusted_score"],
            output_text,
        )

    def evaluate_video_structured(self, user_prompt: str, video_path: str, kwargs: dict):
        if not os.path.exists(video_path):
            raise ValueError(f"not exist: {video_path}")

        max_tokens = kwargs.get("max_tokens", 4096)
        infer_fps = kwargs.get("infer_fps", 2.0)
        temperature = kwargs.get("temperature", 0.7)
        if infer_fps == "raw":
            infer_fps = _get_video_fps(video_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": infer_fps,
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            raise ValueError(f"error when reading: {video_path}")

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

        pattern = (
            r"visual quality:\s*(\d+).*?"
            r"text-to-video alignment:\s*(\d+).*?"
            r"physical/common-sense consistency:\s*(\d+)"
        )
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            hard_scores = {
                "v": int(match.group(1)),
                "t": int(match.group(2)),
                "p": int(match.group(3)),
            }
        else:
            hard_scores = {"v": None, "t": None, "p": None}

        def find_score_token_index_by_prompt(prompt_text: str):
            gen_str = self.tokenizer.decode(gen_token_ids, skip_special_tokens=False)
            pattern = r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(prompt_text)
            match_ = re.search(pattern, gen_str, flags=re.IGNORECASE)
            if not match_:
                return -1

            after_text = gen_str[match_.end():]
            num_match = re.search(r"\d", after_text)
            if not num_match:
                return -1

            target_substr = gen_str[: match_.end() + num_match.start() + 1]
            for i in range(len(gen_token_ids)):
                partial = self.tokenizer.decode(gen_token_ids[: i + 1], skip_special_tokens=False)
                if partial == target_substr:
                    return i
            return -1

        def get_score_probs(token_idx):
            if token_idx < 0:
                return []

            logits = scores[token_idx][0]
            log_probs = torch.log_softmax(logits, dim=-1)
            score_probs = []
            for score in range(1, 6):
                ids = self.tokenizer.encode(str(score), add_special_tokens=False)
                if len(ids) != 1:
                    continue
                prob = float(np.exp(log_probs[ids[0]].item()))
                score_probs.append((score, prob))
            return score_probs

        def compute_metrics(hard_score, token_idx):
            score_probs = get_score_probs(token_idx)
            if hard_score is None or not score_probs:
                return {
                    "expected_score": None,
                    "hard_score": hard_score,
                    "confidence": None,
                    "confidence_adjusted_score": None,
                }

            _, probs = zip(*score_probs)
            total_prob = sum(probs)
            if total_prob <= 0:
                return {
                    "expected_score": None,
                    "hard_score": hard_score,
                    "confidence": None,
                    "confidence_adjusted_score": None,
                }

            norm_probs = [(score, prob / total_prob) for score, prob in score_probs]
            expected_score = round(sum(score * prob for score, prob in norm_probs), 4)
            confidence = round(max(prob for _, prob in norm_probs), 4)
            best_score = max(norm_probs, key=lambda x: x[1])[0]
            confidence_adjusted_score = round(best_score * confidence, 4)
            return {
                "expected_score": expected_score,
                "hard_score": hard_score,
                "confidence": confidence,
                "confidence_adjusted_score": confidence_adjusted_score,
            }

        idx_v = find_score_token_index_by_prompt("visual quality:")
        idx_t = find_score_token_index_by_prompt("text-to-video alignment:")
        idx_p = find_score_token_index_by_prompt("physical/common-sense consistency:")

        v_metrics = compute_metrics(hard_scores["v"], idx_v)
        t_metrics = compute_metrics(hard_scores["t"], idx_t)
        p_metrics = compute_metrics(hard_scores["p"], idx_p)

        structured_metrics = {
            "v_expected_score": v_metrics["expected_score"],
            "v_hard_score": v_metrics["hard_score"],
            "v_confidence": v_metrics["confidence"],
            "v_confidence_adjusted_score": v_metrics["confidence_adjusted_score"],
            "t_expected_score": t_metrics["expected_score"],
            "t_hard_score": t_metrics["hard_score"],
            "t_confidence": t_metrics["confidence"],
            "t_confidence_adjusted_score": t_metrics["confidence_adjusted_score"],
            "p_expected_score": p_metrics["expected_score"],
            "p_hard_score": p_metrics["hard_score"],
            "p_confidence": p_metrics["confidence"],
            "p_confidence_adjusted_score": p_metrics["confidence_adjusted_score"],
        }

        return structured_metrics, output_text
