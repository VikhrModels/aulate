import os
import random
import sys

sys.path.append("BigCodec")
sys.path.append("WavTokenizer")

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

import torch
import torchaudio

from tqdm import tqdm

from pywer import wer, cer

from base import Evaluator


@dataclass
class AudioMetricsResult:
    cer: float
    wer: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ASREvaluator(Evaluator):
    def decode_asr(self, audio, text_tokens=None, quantizer=None):
        """Default implementation of decode_tts"""
        if self._custom_decode_fn:
            return self._custom_decode_fn(audio)

        quantizer = quantizer or self.quantizer
        tokens = quantizer.encode(audio, text_tokens)

        return tokens

    def infer_audio_to_text(
        self,
        audio,
        prompt: str = None,
        do_sample=True,
        top_k: int = 20,
        temperature=0.2,
        top_p=0.99,
        **kwargs,
    ):
        if self._custom_inference_fn:
            return self._custom_inference_fn(audio, prompt)

        text_tokens = (
            self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
            if prompt is not None
            else None
        )
        tokens = self.decode_asr(audio, text_tokens)
        attention_mask = torch.ones(tokens.size(), device=self.device)

        # print("Top-p: ", top_p, "Top-k: ", top_k, "Temperature: ", temperature)

        output_text_tokens = self.model.generate(
            tokens,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,
            eos_token_id=self.end_sequence_token_id,
            **kwargs,
        )
        output_text_tokens = output_text_tokens.cpu()[0]
        if text_tokens is not None:
            output_text_tokens = output_text_tokens[len(text_tokens[0]) + 1 :]

        decoded_text = self.tokenizer.decode(
            output_text_tokens, skip_special_tokens=True
        )
        # import pdb; pdb.set_trace()

        # if prompt is not None:
        #     return decoded_text(prompt)
        return decoded_text

    def calculate_metrics(
        self,
        reference_text: str,
        generated_text: str,
    ) -> Optional[AudioMetricsResult]:
        """Calculate C47306ER, WER metrics"""
        reference_text = reference_text.lower().strip(".")
        generated_text = generated_text.lower().strip(".")

        # Calculate standard metrics
        try:
            cer_score = cer([reference_text], [generated_text])
        except Exception as e:
            print(f"CER calculation failed: {str(e)}")
            cer_score = 100.0

        try:
            wer_score = wer([reference_text], [generated_text])
        except Exception as e:
            print(f"WER calculation failed: {str(e)}")
            wer_score = 100.0

        return AudioMetricsResult(
            cer=float(cer_score),
            wer=float(wer_score),
        )

    def evaluate_batch(
        self,
        samples: list[tuple[str, dict]],
        prompt: Optional[str] = None,
        batch_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Evaluate metrics on a batch of samples.

        Args:
            samples: List of (text, reference_audio) tuples or just reference_audio if no text
            prompt: Optional prompt to use for text-to-audio generation
            batch_metadata: Optional metadata to include in results

        Returns:
            DataFrame containing evaluation results
        """
        results = []
        for idx, sample in tqdm(
            enumerate(samples), total=len(samples), desc="Processing asr samples"
        ):
            try:
                text, audio = sample
                prediction = self.infer_audio_to_text(audio, prompt, **kwargs)
                metrics = self.calculate_metrics(text, prediction)

                print({"CER": metrics.cer, "WER": metrics.wer})
                print("Prediction: ", prediction)
                print("Reference: ", text)
                result_dict = {"CER": metrics.cer, "WER": metrics.wer}

                if batch_metadata:
                    result_dict.update(batch_metadata)

                results.append(result_dict)

            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue

        # Create DataFrame with results
        df = pd.DataFrame(results)

        # Save results
        if len(results) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(
                self.results_dir, f"metrics_evaluation_{timestamp}.csv"
            )
            df.to_csv(csv_path, index=False)

        return df


def evaluate_on_librispeech(
    evaluator: ASREvaluator,
    num_samples: int = 200,
    prompt: str = "Transcribe the audio. ",
    subset: str = "test-clean",
    random_seed: int = 42,
    **kwargs,
):
    """Evaluate metrics on LibriSpeech samples"""
    print(f"Loading LibriSpeech dataset ({subset})...")
    dataset = torchaudio.datasets.LIBRISPEECH("./data", url=subset, download=True)

    # Randomly sample entries
    all_indices = list(range(len(dataset)))
    random.seed(random_seed)
    random.shuffle(all_indices)
    indices = all_indices[:num_samples]

    samples = []
    metadata = []
    print("Preparing samples...")
    for idx in tqdm(indices, desc="Loading audio files"):
        waveform, sample_rate, text, _, _, _ = dataset[torch.tensor(idx).long()]
        reference_audio = waveform.numpy().squeeze()
        samples.append((text, {"array": reference_audio, "sampling_rate": sample_rate}))

    print("Running evaluation...")
    results_df = evaluator.evaluate_batch(
        samples,
        prompt=prompt,
        batch_metadata={"dataset": "librispeech", "subset": subset},
        **kwargs,
    )

    # Add metadata
    for idx, meta in enumerate(metadata):
        for key, value in meta.items():
            results_df.loc[results_df["sample_idx"] == idx, key] = value

    # Calculate and print average metrics
    avg_metrics = results_df[["CER", "WER"]].mean()
    std_metrics = results_df[["CER", "WER"]].std()

    print("\nMetrics Summary:")
    for metric in avg_metrics.index:
        print(
            f"{metric}: Mean = {avg_metrics[metric]:.4f}, Std = {std_metrics[metric]:.4f}"
        )

    return results_df


if __name__ == "__main__":
    asr_conf = {"type": "speech", "n_tokens_before": 8192, "kwargs": {}}
    tts_conf = {"type": "bigcodec", "kwargs": {}}

    evaluator = ASREvaluator(
        base_model="ksych/salt-asr-tts-99k",
        audio_tokenizer_config={"asr": asr_conf, "tts": tts_conf},
    )

    results_df = evaluate_on_librispeech(
        evaluator=evaluator,
        num_samples=500,
        # prompt=None,
        prompt="Transcribe the audio.",
        # do_sample=False,
        # num_beams=5,
        # early_stopping=True,
        # length_penalty=1.5,
    )
