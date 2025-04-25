import os
import sys
from abc import abstractmethod

sys.path.append("BigCodec")
sys.path.append("WavTokenizer")

from typing import Dict, Any, Callable, Optional

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tokenizer import (
    SpeechAudioTokenizer,
    BigCodecsAudioTokenizer,
    MixedAudioTokenizer,
    WavAudioTokenizer,
)


class Evaluator:
    def __init__(
        self,
        base_model: Optional[str] = None,
        audio_tokenizer_config: Optional[dict] = None,
        decode_fn: Optional[Callable] = None,
        inference_fn: Optional[Callable] = None,
        device: str = "cuda",
        cache_dir: str = ".",
        results_dir: str = "evaluation_results",
        token: Optional[str] = None,
    ):
        """
        Initialize the AudioMetricsEvaluator.

        Args:
            base_model: Path to the base model (optional if custom decode_fn and inference_fn are provided)
            audio_tokenizer_config: Pretrained audio tokenizer (optional if custom decode_fn and inference_fn are provided)
            decode_fn: Custom function to decode audio tokens to waveform (optional)
            inference_fn: Custom function to generate audio from text (optional)
            device: Device to run computations on
            cache_dir: Directory for model caching
            results_dir: Directory to save evaluation results
        """
        self.device = device
        self.results_dir = results_dir
        self.cache_dir = cache_dir
        os.makedirs(results_dir, exist_ok=True)

        # Initialize custom functions if provided
        self._custom_decode_fn = decode_fn
        self._custom_inference_fn = inference_fn
        self.token = token

        self.start_audio_token = "<|start_of_audio|>"
        self.end_audio_token = "<|end_of_audio|>"
        self.start_sequence_token = "<|im_start|>"
        self.end_sequence_token = "<|im_end|>"

        if not (decode_fn and inference_fn):
            self._initialize_default_implementation(
                base_model,
                audio_tokenizer_config,
            )

    def _load_audio_tokenizer(self, audio_tokenizer_config):
        asr_config = audio_tokenizer_config["asr"]
        tts_config = audio_tokenizer_config["tts"]

        n_tokens = len(self.tokenizer)
        start_audio_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.start_audio_token)]],
            device=self.device,
        )
        end_audio_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.end_audio_token)]],
            device=self.device,
        )
        start_sequence_token_id = torch.tensor(
            [[self.tokenizer.convert_tokens_to_ids(self.start_sequence_token)]],
            device=self.device,
        )

        if asr_config["type"] == "speech":
            asr = SpeechAudioTokenizer(
                **asr_config["kwargs"],
                n_original_tokens=n_tokens,
                start_audio_token_id=start_audio_token_id,
                end_audio_token_id=end_audio_token_id,
                start_sequence_token_id=start_sequence_token_id,
            )
        elif asr_config["type"] == "bigcodec":
            asr = BigCodecsAudioTokenizer(
                **asr_config["kwargs"],
                n_original_tokens=n_tokens,
                start_audio_token_id=start_audio_token_id,
                end_audio_token_id=end_audio_token_id,
                start_sequence_token_id=start_sequence_token_id,
            )
        elif asr_config["type"] == "wav":
            asr = WavAudioTokenizer(
                **asr_config["kwargs"],
                n_original_tokens=n_tokens,
                start_audio_token_id=start_audio_token_id,
                end_audio_token_id=end_audio_token_id,
                start_sequence_token_id=start_sequence_token_id,
            )
        else:
            raise ValueError(f"Invalid audio tokenizer type: {asr_config['type']}")

        if tts_config["type"] == "speech":
            tts = SpeechAudioTokenizer(
                **tts_config["kwargs"],
                n_original_tokens=n_tokens,
                start_audio_token_id=start_audio_token_id,
                end_audio_token_id=end_audio_token_id,
            )
        elif tts_config["type"] == "bigcodec":
            tts = BigCodecsAudioTokenizer(
                **tts_config["kwargs"],
                n_original_tokens=n_tokens,
                start_audio_token_id=start_audio_token_id,
                end_audio_token_id=end_audio_token_id,
            )
        elif tts_config["type"] == "wav":
            tts = WavAudioTokenizer(
                **tts_config["kwargs"],
                n_original_tokens=n_tokens,
                start_audio_token_id=start_audio_token_id,
                end_audio_token_id=end_audio_token_id,
            )
        else:
            raise ValueError(f"Invalid audio tokenizer type: {tts_config['type']}")

        return MixedAudioTokenizer(asr, tts, n_original_tokens=n_tokens)

    def _initialize_default_implementation(self, base_model, audio_tokenizer_config):
        """Initialize the default model implementation"""
        if not all([base_model, audio_tokenizer_config]):
            return

        self.start_audio_token = "<|start_of_audio|>"
        self.end_audio_token = "<|end_of_audio|>"
        self.start_sequence_token = "<|im_start|>"
        self.end_sequence_token = "<|im_end|>"

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, cache_dir=self.cache_dir, use_auth_token=self.token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": 0},
            use_auth_token=self.token,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.end_sequence_token_id = self.tokenizer.convert_tokens_to_ids(
            self.end_sequence_token
        )

        self.quantizer = self._load_audio_tokenizer(audio_tokenizer_config)

    def set_decode_fn(self, decode_fn: Callable) -> None:
        """Set a custom function to decode audio tokens to waveform."""
        self._custom_decode_fn = decode_fn

    def set_inference_fn(self, inference_fn: Callable) -> None:
        """Set a custom function to generate audio from text."""
        self._custom_inference_fn = inference_fn

    @abstractmethod
    def calculate_metrics(
        self,
        reference_text: str,
        generated_text: str,
    ):
        pass

    @abstractmethod
    def evaluate_batch(
        self,
        samples: list[tuple[str, dict]],
        prompt: Optional[str] = None,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        pass
