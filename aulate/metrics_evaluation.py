import numpy as np
import os
import sys
sys.path.append("WavTokenizer")
from pesq import pesq
from pystoi import stoi
from .si_sdr import si_sdr
import soundfile as sf
from torch_mir_eval import bss_eval_sources
import torchaudio
import random
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from speechtokenizer import SpeechTokenizer
from WavTokenizer.decoder.pretrained import WavTokenizer
from audiotools import AudioSignal
from datetime import datetime
import librosa
from typing import Dict, Any, Callable, Optional, Union
from dataclasses import dataclass

# pip install pesq pystoi torch-mir-eval torchaudio pandas torch transformers soundfile tqdm numpy git+https://github.com/descriptinc/audiotools

@dataclass
class AudioMetricsResult:
    pesq: float
    stoi: float
    si_sdr: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AudioMetricsEvaluator:
    def __init__(
        self,
        base_model: Optional[str] = None,
        speechtokenizer_config: Optional[str] = None,
        speechtokenizer_checkpoint: Optional[str] = None,
        decode_fn: Optional[Callable] = None,
        inference_fn: Optional[Callable] = None,
        device: str = "cuda",
        cache_dir: str = ".",
        results_dir: str = "evaluation_results"
    ):
        """
        Initialize the AudioMetricsEvaluator.

        Args:
            base_model: Path to the base model (optional if custom decode_fn and inference_fn are provided)
            speechtokenizer_config: Path to speechtokenizer config (optional if custom functions provided)
            speechtokenizer_checkpoint: Path to speechtokenizer checkpoint (optional if custom functions provided)
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

        # Initialize default implementation if no custom functions provided
        if not (decode_fn and inference_fn):
            self._initialize_default_implementation(
                base_model,
                speechtokenizer_config,
                speechtokenizer_checkpoint
            )

        self.gen_audio_dir = "gen_a"
        os.makedirs(self.gen_audio_dir, exist_ok=True)

    def _initialize_default_implementation(self, base_model, speechtokenizer_config, speechtokenizer_checkpoint):
        """Initialize the default model implementation"""
        if not all([base_model, speechtokenizer_config, speechtokenizer_checkpoint]):
            return

        self.start_audio_token = "<|start_of_audio|>"
        self.end_audio_token = "<|end_of_audio|>"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": 0}
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Load quantizer
        self.quantizer = WavTokenizer.from_pretrained0802(
            "WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
            "audiotokenizer/wavtokenizer_large_unify_600_24k.ckpt"
        )
        self.quantizer = self.quantizer.to(self.model.device)

    def set_decode_fn(self, decode_fn: Callable) -> None:
        """Set a custom function to decode audio tokens to waveform."""
        self._custom_decode_fn = decode_fn

    def set_inference_fn(self, inference_fn: Callable) -> None:
        """Set a custom function to generate audio from text."""
        self._custom_inference_fn = inference_fn

    def decode_tts(self, tokens, quantizer=None, n_codebooks=1, n_original_tokens=None, start_audio_token_id=None, end_audio_token_id=None):
        """Default implementation of decode_tts"""
        if self._custom_decode_fn:
            return self._custom_decode_fn(tokens)

        # Use default implementation
        quantizer = quantizer or self.quantizer
        n_original_tokens = n_original_tokens or len(self.tokenizer)
        start_audio_token_id = start_audio_token_id or self.tokenizer(self.start_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(self.device)
        end_audio_token_id = end_audio_token_id or self.tokenizer(self.end_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(self.device)

        tokens = tokens % n_original_tokens
        start = torch.nonzero(tokens == start_audio_token_id)
        end = torch.nonzero(tokens == end_audio_token_id)

        start = start[0, -1] + 1 if len(start) else 0
        end = end[0, -1] if len(end) else tokens.shape[-1]

        audio_tokens = tokens[start:end]
        transposed = audio_tokens.view(-1, n_codebooks).t()
        codes = transposed.view(n_codebooks, 1, -1).to(self.device)

        features = quantizer.codes_to_features(codes)
        bandwidth_id = torch.tensor([0], device=self.device)

        audio = quantizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)

        del tokens
        del audio_tokens
        torch.cuda.empty_cache()

        return AudioSignal(audio.detach().cpu().numpy(), 24000)

    def infer_text_to_audio(self, text: str, prompt: str, top_k: int = 50):
        """Default implementation of text-to-audio inference"""
        if self._custom_inference_fn:
            return self._custom_inference_fn(text, prompt)

        # Use default implementation
        max_seq_length = 1024
        formatted_text = f"Say '{text.upper()}' {prompt}"

        text_tokenized = self.tokenizer(formatted_text, return_tensors="pt")
        text_input_tokens = text_tokenized["input_ids"].to(self.device)

        soa = self.tokenizer(self.start_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(self.device)
        eoa = self.tokenizer(self.end_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(self.device)

        text_tokens = torch.cat([text_input_tokens, soa], dim=1)
        attention_mask = torch.ones(text_tokens.size(), device=self.device)

        output_audio_tokens = self.model.generate(
            text_tokens,
            attention_mask=attention_mask,
            max_new_tokens=max_seq_length,
            top_k=top_k,
            do_sample=True,
            temperature=0.5,
            repetition_penalty=1.1,
            length_penalty=1.2,
        )

        return self.decode_tts(output_audio_tokens[0], self.quantizer, 1, len(self.tokenizer), soa, eoa)

    def si_sdr(self, reference: torch.Tensor, estimation: torch.Tensor) -> float:
        """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)"""
        reference = reference.to(self.device)
        estimation = estimation.to(self.device)

        reference = reference - torch.mean(reference)
        estimation = estimation - torch.mean(estimation)

        alpha = torch.sum(reference * estimation) / torch.sum(reference ** 2)
        scaled_reference = alpha * reference

        noise = estimation - scaled_reference
        ratio = torch.sum(scaled_reference ** 2) / (torch.sum(noise ** 2) + 1e-8)
        si_sdr_value = 10 * torch.log10(ratio + 1e-8)

        return float(si_sdr_value.item())

    def calculate_metrics(
        self,
        reference_audio: Union[np.ndarray, torch.Tensor],
        generated_audio: Union[np.ndarray, torch.Tensor],
        sr: int = 24000
    ) -> AudioMetricsResult:
        """Calculate PESQ, STOI and SI-SDR metrics"""
        # Convert to numpy if needed
        if isinstance(reference_audio, torch.Tensor):
            reference_audio = reference_audio.detach().cpu().numpy()
        if isinstance(generated_audio, torch.Tensor):
            generated_audio = generated_audio.detach().cpu().numpy()

        # Ensure same length
        min_len = min(len(reference_audio), len(generated_audio))
        reference_audio = reference_audio[:min_len]
        generated_audio = generated_audio[:min_len]

        # Normalize audio
        reference_audio = reference_audio / np.max(np.abs(reference_audio))
        generated_audio = generated_audio / np.max(np.abs(generated_audio))

        # Resample to 16kHz for PESQ
        reference_audio_16k = librosa.resample(reference_audio, orig_sr=sr, target_sr=16000)
        generated_audio_16k = librosa.resample(generated_audio, orig_sr=sr, target_sr=16000)

        # Calculate metrics
        pesq_score = pesq(16000, reference_audio_16k, generated_audio_16k, 'wb')
        stoi_score = stoi(reference_audio, generated_audio, sr, extended=False)
        sisdr_score = si_sdr(np.array(reference_audio), np.array(generated_audio))

        return AudioMetricsResult(
            pesq=pesq_score,
            stoi=stoi_score,
            si_sdr=sisdr_score
        )

    def evaluate_batch(
        self,
        samples: list,
        prompt: Optional[str] = None,
        batch_metadata: Optional[Dict[str, Any]] = None,
        save_audio: bool = False
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
        for idx, sample in enumerate(samples):
            try:
                if isinstance(sample, tuple):
                    text, reference_audio = sample
                    generated_audio = self.infer_text_to_audio(text, prompt) if prompt else self.infer_text_to_audio(text, "")
                    if isinstance(generated_audio, AudioSignal):
                        generated_audio = generated_audio.audio_data.squeeze()
                else:
                    reference_audio = sample
                    generated_audio = self.decode_tts(reference_audio)
                    if isinstance(generated_audio, AudioSignal):
                        generated_audio = generated_audio.audio_data.squeeze()
                    text = None

                metrics = self.calculate_metrics(reference_audio, generated_audio)

                result_dict = {
                    'PESQ': metrics.pesq,
                    'STOI': metrics.stoi,
                    'SI-SDR': metrics.si_sdr,
                    'sample_idx': idx
                }

                if text:
                    result_dict['text'] = text
                if batch_metadata:
                    result_dict.update(batch_metadata)

                if save_audio:
                    audio_filename = f"gen_{idx}.wav"
                    audio_path = os.path.join(self.gen_audio_dir, audio_filename)
                    sf.write(audio_path, generated_audio, 24000)
                    result_dict['audio_path'] = audio_path

                results.append(result_dict)

            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue

        # Create DataFrame with results
        df = pd.DataFrame(results)

        # Save results
        if len(results) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(self.results_dir, f'metrics_evaluation_{timestamp}.csv')
            df.to_csv(csv_path, index=False)

        return df

    def evaluate_on_librispeech(
        self,
        num_samples: int = 200,
        prompt: str = "with a natural speaking voice, clear pronunciation, and minimal background noise",
        subset: str = "test-clean",
        save_audio: bool = False
    ):
        """Evaluate metrics on LibriSpeech samples"""
        dataset = torchaudio.datasets.LIBRISPEECH("./data", url=subset, download=True)

        indices = random.sample(range(len(dataset)), num_samples)

        samples = []
        metadata = []
        for idx in indices:
            waveform, sample_rate, text, speaker_id, chapter_id, utterance_id = dataset[torch.tensor(idx).long()]
            reference_audio = waveform.numpy().squeeze()
            if sample_rate != 24000:
                reference_audio = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=24000)
            samples.append((text, reference_audio))
            metadata.append({
                'speaker_id': speaker_id,
                'chapter_id': chapter_id,
                'utterance_id': utterance_id
            })

        results_df = self.evaluate_batch(
            samples,
            prompt=prompt,
            batch_metadata={'dataset': 'librispeech', 'subset': subset},
            save_audio=save_audio
        )

        for idx, meta in enumerate(metadata):
            for key, value in meta.items():
                results_df.loc[results_df['sample_idx'] == idx, key] = value

        avg_metrics = results_df[['PESQ', 'STOI', 'SI-SDR']].mean()
        print("\nAverage Metrics:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")

        return results_df

if __name__ == "__main__":
    evaluator = AudioMetricsEvaluator(
        base_model='/workspace/models--ksych--salt-asr_wav-uni_1_tts_wav-uni_1-12k/snapshots/1214eda33861945eb09163f97f2f29284070dde4',
        speechtokenizer_config="audiotokenizer/speechtokenizer_hubert_avg_config.json",
        speechtokenizer_checkpoint="audiotokenizer/SpeechTokenizer.pt"
    )

    results_df = evaluator.evaluate_on_librispeech(
        num_samples=50,
        prompt="with a male speaker delivers a very monotone and high-pitched speech with a very fast speed in a setting with almost no noise, creating a clear and loud recording.",
    )

