import os
import sys
import random
import argparse

from transformers import AutoModelForCausalLM
from vllm import SamplingParams, LLM

sys.path.append("BigCodec")
sys.path.append("WavTokenizer")

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd

import soundfile as sf

import torch
import torchaudio

from tqdm import tqdm
from datasets import load_dataset

import librosa
from audiotools import AudioSignal
from torchaudio.pipelines import SQUIM_OBJECTIVE

from pesq import pesq
from pystoi import stoi
from si_sdr import si_sdr

from base import Evaluator, set_seed


@dataclass
class AudioMetricsResult:
    pesq: float
    stoi: float
    si_sdr: float
    sim_o: float  # Overall similarity
    sim_r: float  # Rhythm similarity
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TTSEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gen_audio_dir = "gen_a"
        os.makedirs(self.gen_audio_dir, exist_ok=True)

    def _load_model(self, base_model: str):
        if self.use_vllm:
            self.model = LLM(base_model, gpu_memory_utilization=0.5)
        else:
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

    def decode_tts(self, tokens, quantizer=None):
        """Default implementation of decode_tts"""
        if self._custom_decode_fn:
            return self._custom_decode_fn(tokens)

        quantizer = quantizer or self.quantizer
        return quantizer.decode(tokens)

    def infer_text_to_audio(
        self,
        text: str,
        prompt: str,
        top_k: int = 200,
        temperature: float = 1.2,
        do_sample: bool = True,
        **kwargs,
    ):
        """Default implementation of text-to-audio inference"""
        if self._custom_inference_fn:
            return self._custom_inference_fn(text, prompt)

        # Use default implementation
        max_seq_length = 1024
        formatted_text = text.lower()

        text_tokenized = self.tokenizer(formatted_text, return_tensors="pt")
        text_input_tokens = text_tokenized["input_ids"].to(self.device)

        text_tokens = torch.cat(
            [
                self.start_sequence_token_id,
                text_input_tokens,
                self.start_audio_token_id,
            ],
            dim=1,
        )

        if self.use_vllm:
            sampling_params = SamplingParams(
                max_tokens=max_seq_length,
                top_k=top_k,
                temperature=temperature if do_sample else 0.0,
                stop_token_ids=[self.end_audio_token_id.cpu().item()],
            )
            # import pdb; pdb.set_trace()
            # print(self.tokenizer.decode(text_tokens)[0])
            output_audio_tokens = (
                self.model.generate(
                    self.tokenizer.decode(text_tokens[0]),
                    sampling_params=sampling_params,
                )[0]
                .outputs[0]
                .token_ids
            )
            output_audio_tokens = torch.tensor(output_audio_tokens, device=self.device)
        else:
            attention_mask = torch.ones(text_tokens.size(), device=self.device)

            output_audio_tokens = self.model.generate(
                text_tokens,
                attention_mask=attention_mask,
                max_new_tokens=max_seq_length,
                top_k=top_k,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.end_audio_token_id,
                # repetition_penalty=1.1,
                **kwargs,
            )[0]

        return self.decode_tts(output_audio_tokens, self.quantizer)

    def si_sdr(self, reference: torch.Tensor, estimation: torch.Tensor) -> float:
        return si_sdr(
            reference.detach().cpu().numpy(), estimation.detach().cpu().numpy()
        )

    def calculate_metrics(
        self,
        reference_audio: Union[np.ndarray, torch.Tensor],
        generated_audio: Union[np.ndarray, torch.Tensor],
        ref_sr: int = 16000,
        gen_sr: int = 44100,
    ) -> Optional[AudioMetricsResult]:
        """Calculate PESQ, STOI, SI-SDR, SIM-O, and SIM-R metrics"""
        # Resample reference and generated audio to a common sample rate for SI-SDR calculation
        if isinstance(reference_audio, torch.Tensor):
            reference_audio = reference_audio.detach().cpu().numpy()
        if isinstance(generated_audio, torch.Tensor):
            generated_audio = generated_audio.detach().cpu().numpy()

        min_len = min(len(reference_audio), len(generated_audio))
        reference_audio = reference_audio[:min_len]
        generated_audio = generated_audio[:min_len]
        target_sr = SQUIM_OBJECTIVE.sample_rate
        if ref_sr != target_sr:
            reference_audio = librosa.resample(
                reference_audio.astype(np.float32), orig_sr=ref_sr, target_sr=target_sr
            )
        if gen_sr != target_sr:
            generated_audio = librosa.resample(
                generated_audio.astype(np.float32), orig_sr=gen_sr, target_sr=target_sr
            )

        # Convert to numpy and ensure float32
        reference_audio = reference_audio.astype(np.float32)
        generated_audio = generated_audio.astype(np.float32)

        # Convert to torch tensors for SI-SDR calculation
        reference_tensor = torch.tensor(reference_audio)[None, :].to(
            self.device, dtype=torch.float32
        )
        generated_tensor = torch.tensor(generated_audio)[None, :].to(
            self.device, dtype=torch.float32
        )

        # Ensure same length
        min_len = min(reference_tensor.shape[1], generated_tensor.shape[1])
        reference_tensor = reference_tensor[:, :min_len]
        generated_tensor = generated_tensor[:, :min_len]

        # Calculate SI-SDR using SQUIM
        try:
            max_audio_length = 15 * target_sr
            model = SQUIM_OBJECTIVE.get_model().to(self.device)
            with torch.no_grad():
                reference_tensor = reference_tensor[
                    :, : min(max_audio_length, reference_tensor.shape[1])
                ]
                generated_tensor = generated_tensor[
                    :, : min(max_audio_length, generated_tensor.shape[1])
                ]
                _, _, sisdr_score = model(generated_tensor)
                sisdr_score = sisdr_score.cpu()[0]
        except Exception as e:
            print(f"SI-SDR calculation failed: {str(e)}")
            sisdr_score = float("-inf")
        finally:
            if "model" in locals():
                model.cpu()
                del model
                torch.cuda.empty_cache()

        # Resample reference and generated audio to 16kHz for PESQ
        reference_audio_16k = librosa.resample(
            reference_audio, orig_sr=target_sr, target_sr=16000
        )
        generated_audio_16k = librosa.resample(
            generated_audio, orig_sr=target_sr, target_sr=16000
        )

        # Resample to 24kHz for STOI (as it works better with this sample rate)
        reference_audio_24k = librosa.resample(
            reference_audio, orig_sr=target_sr, target_sr=24000
        )
        generated_audio_24k = librosa.resample(
            generated_audio, orig_sr=target_sr, target_sr=24000
        )

        # Calculate standard metrics
        try:
            pesq_score = pesq(16000, reference_audio_16k, generated_audio_16k, "wb")
        except Exception as e:
            print(f"PESQ calculation failed: {str(e)}")
            pesq_score = -0.5

        try:
            stoi_score = stoi(
                reference_audio_24k[
                    : min(len(reference_audio_24k), len(generated_audio_24k))
                ],
                generated_audio_24k[
                    : min(len(reference_audio_24k), len(generated_audio_24k))
                ],
                24000,
                extended=False,
            )
        except Exception as e:
            print(f"STOI calculation failed: {str(e)}")
            stoi_score = -1.0

        # Calculate SIM-O (Overall Similarity)
        try:
            # Using MFCCs for overall spectral similarity
            mfcc_ref = librosa.feature.mfcc(y=reference_audio_24k, sr=24000, n_mfcc=13)
            mfcc_gen = librosa.feature.mfcc(y=generated_audio_24k, sr=24000, n_mfcc=13)

            # Normalize MFCCs
            mfcc_ref = (mfcc_ref - np.mean(mfcc_ref)) / (np.std(mfcc_ref) + 1e-8)
            mfcc_gen = (mfcc_gen - np.mean(mfcc_gen)) / (np.std(mfcc_gen) + 1e-8)

            # Calculate cosine similarity for each frame and take mean
            sim_o = np.mean(
                [
                    np.dot(mfcc_ref[:, i], mfcc_gen[:, i])
                    / (
                        np.linalg.norm(mfcc_ref[:, i]) * np.linalg.norm(mfcc_gen[:, i])
                        + 1e-8
                    )
                    for i in range(min(mfcc_ref.shape[1], mfcc_gen.shape[1]))
                ]
            )
        except Exception as e:
            print(f"SIM-O calculation failed: {str(e)}")
            sim_o = 0.0

        # Calculate SIM-R (Rhythm Similarity)
        try:
            # Using onset strength envelope
            onset_env_ref = librosa.onset.onset_strength(
                y=reference_audio_24k, sr=24000
            )
            onset_env_gen = librosa.onset.onset_strength(
                y=generated_audio_24k, sr=24000
            )

            # Normalize onset envelopes
            onset_env_ref = (onset_env_ref - np.mean(onset_env_ref)) / (
                np.std(onset_env_ref) + 1e-8
            )
            onset_env_gen = (onset_env_gen - np.mean(onset_env_gen)) / (
                np.std(onset_env_gen) + 1e-8
            )

            # Calculate correlation coefficient
            sim_r = np.corrcoef(
                onset_env_ref[: min(len(onset_env_ref), len(onset_env_gen))],
                onset_env_gen[: min(len(onset_env_ref), len(onset_env_gen))],
            )[0, 1]
        except Exception as e:
            print(f"SIM-R calculation failed: {str(e)}")
            sim_r = -1.0

        return AudioMetricsResult(
            pesq=float(pesq_score),
            stoi=float(stoi_score),
            si_sdr=float(sisdr_score),
            sim_o=float(sim_o),
            sim_r=float(sim_r),
        )

    def evaluate_batch(
        self,
        samples: list,
        prompt: Optional[str] = None,
        batch_metadata: Optional[Dict[str, Any]] = None,
        save_audio: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate metrics on a batch of samples.

        Args:
            samples: List of (text, reference_audio) tuples or just reference_audio if no text
            prompt: Optional prompt to use for text-to-audio generation
            batch_metadata: Optional metadata to include in results
            save_audio: Whether to save generated audio files

        Returns:
            DataFrame containing evaluation results
        """
        results = []
        for idx, sample in tqdm(
            enumerate(samples), total=len(samples), desc="Processing samples"
        ):
            try:
                if isinstance(sample, tuple):
                    text, reference_audio, ref_sr = sample
                    print(text, prompt)
                    generated_audio, sr = (
                        self.infer_text_to_audio(text, prompt)
                        if prompt
                        else self.infer_text_to_audio(text, "")
                    )
                    if isinstance(generated_audio, AudioSignal):
                        generated_audio = generated_audio.audio_data.squeeze()
                else:
                    reference_audio = sample
                    generated_audio = self.decode_tts(reference_audio)
                    if isinstance(generated_audio, AudioSignal):
                        generated_audio = generated_audio.audio_data.squeeze()
                    text = None

                metrics = self.calculate_metrics(
                    reference_audio, generated_audio, ref_sr=ref_sr, gen_sr=sr
                )
                print(
                    {
                        "PESQ": metrics.pesq,
                        "STOI": metrics.stoi,
                        "SI-SDR": metrics.si_sdr,
                        "SIM-O": metrics.sim_o,
                        "SIM-R": metrics.sim_r,
                    }
                )

                result_dict = {
                    "PESQ": metrics.pesq,
                    "STOI": metrics.stoi,
                    "SI-SDR": metrics.si_sdr,
                    "SIM-O": metrics.sim_o,
                    "SIM-R": metrics.sim_r,
                    "sample_idx": idx,
                }

                if text:
                    result_dict["text"] = text
                if batch_metadata:
                    result_dict.update(batch_metadata)

                if save_audio:
                    audio_filename = f"gen_{idx}.wav"
                    audio_filename_ = f"orig_{idx}.wav"
                    audio_path = os.path.join(self.gen_audio_dir, audio_filename)
                    audio_path_ = os.path.join(self.gen_audio_dir, audio_filename_)
                    sf.write(audio_path, generated_audio.astype(np.float32), sr)
                    sf.write(audio_path_, reference_audio.astype(np.float32), ref_sr)
                    result_dict["audio_path"] = audio_path

                results.append(result_dict)

            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                result_dict = {
                    "PESQ": -0.5,
                    "STOI": -1.0,
                    "SI-SDR": -50,
                    "SIM-O": 0.0,
                    "SIM-R": -1.0,
                    "sample_idx": idx,
                }
                results.append(result_dict)
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
    evaluator: TTSEvaluator,
    num_samples: int = 200,
    prompt: Optional[
        str
    ] = "with a natural speaking voice, clear pronunciation, and minimal background noise",
    subset: str = "test-clean",
    save_audio=True,
    random_seed: int = 42,
):
    """Evaluate metrics on LibriSpeech samples"""
    set_seed(random_seed)
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
        waveform, sample_rate, text, speaker_id, chapter_id, utterance_id = dataset[
            torch.tensor(idx).long()
        ]
        reference_audio = waveform.numpy().squeeze()
        samples.append((text, reference_audio))
        metadata.append(
            {
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utterance_id": utterance_id,
            }
        )

    print("Running evaluation...")
    results_df = evaluator.evaluate_batch(
        samples,
        prompt=prompt,
        batch_metadata={"dataset": "librispeech", "subset": subset},
        save_audio=save_audio,
    )

    # Add metadata
    for idx, meta in enumerate(metadata):
        for key, value in meta.items():
            results_df.loc[results_df["sample_idx"] == idx, key] = value

    # Calculate and print average metrics
    avg_metrics = results_df[["PESQ", "STOI", "SI-SDR", "SIM-O", "SIM-R"]].mean()
    std_metrics = results_df[["PESQ", "STOI", "SI-SDR", "SIM-O", "SIM-R"]].std()

    print("\nMetrics Summary:")
    for metric in avg_metrics.index:
        print(
            f"{metric}: Mean = {avg_metrics[metric]:.4f}, Std = {std_metrics[metric]:.4f}"
        )

    return results_df


def evaluate_on_mozilla(
    evaluator: TTSEvaluator,
    num_samples: int = 200,
    prompt: Optional[str] = None,
    subset: str = "ru",
    save_audio=True,
    random_seed: int = 42,
):
    """Evaluate metrics on LibriSpeech samples"""
    set_seed(random_seed)
    print(f"Loading ru mozilla dataset ({subset})...")
    dataset = load_dataset("mozilla-foundation/common_voice_12_0", subset)["test"]

    # Randomly sample entries
    all_indices = list(range(len(dataset)))
    random.seed(random_seed)
    random.shuffle(all_indices)
    indices = all_indices[:num_samples]

    samples = []
    metadata = []
    print("Preparing samples...")
    for idx in tqdm(indices, desc="Loading audio files"):
        row = dataset[idx]
        reference_audio, sample_rate, text = (
            row["audio"]["array"],
            row["audio"]["sampling_rate"],
            row["sentence"],
        )
        samples.append((text, reference_audio, sample_rate))

    print("Running evaluation...")
    results_df = evaluator.evaluate_batch(
        samples,
        prompt=prompt,
        batch_metadata={"dataset": "mozilla", "subset": subset},
        save_audio=save_audio,
    )

    # Add metadata
    for idx, meta in enumerate(metadata):
        for key, value in meta.items():
            results_df.loc[results_df["sample_idx"] == idx, key] = value

    # Calculate and print average metrics
    avg_metrics = results_df[["PESQ", "STOI", "SI-SDR", "SIM-O", "SIM-R"]].mean()
    std_metrics = results_df[["PESQ", "STOI", "SI-SDR", "SIM-O", "SIM-R"]].std()

    print("\nMetrics Summary:")
    for metric in avg_metrics.index:
        print(
            f"{metric}: Mean = {avg_metrics[metric]:.4f}, Std = {std_metrics[metric]:.4f}"
        )

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR model on LibriSpeech")
    parser.add_argument(
        "--model", type=str, default="ksych/salt-audiobooks-last", help="Model name"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate on"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Prompt to prepend to transcription"
    )
    parser.add_argument("--subset", type=str, default="ru", help="Dataset subset")
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed to use."
    )

    args = parser.parse_args()

    asr_conf = {"type": "speech", "kwargs": {}}
    tts_conf = {"type": "bigcodec", "kwargs": {}}

    evaluator = TTSEvaluator(
        base_model=args.model,
        audio_tokenizer_config={"asr": asr_conf, "tts": tts_conf},
    )

    results_df = evaluate_on_mozilla(
        evaluator,
        num_samples=args.num_samples,
        prompt=args.prompt,
        random_seed=args.random_seed,
    )
