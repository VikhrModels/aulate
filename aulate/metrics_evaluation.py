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
from torchaudio.pipelines import SQUIM_OBJECTIVE

# pip install pesq pystoi torch-mir-eval torchaudio pandas torch transformers soundfile tqdm numpy git+https://github.com/descriptinc/audiotools

@dataclass
class AudioMetricsResult:
    pesq: float
    stoi: float
    si_sdr: float
    sim_o: float  # Overall similarity
    sim_r: float  # Rhythm similarity
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
        results_dir: str = "evaluation_results",
        token: Optional[str] = None
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
        self.token = token


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

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=self.cache_dir, use_auth_token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": 0},
            use_auth_token=self.token
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Load quantizer
        self.quantizer = WavTokenizer.from_pretrained0802(speechtokenizer_config, speechtokenizer_checkpoint)
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
        return si_sdr(reference.detach().cpu().numpy(), estimation.detach().cpu().numpy())

    def calculate_metrics(
        self,
        reference_audio: Union[np.ndarray, torch.Tensor],
        generated_audio: Union[np.ndarray, torch.Tensor],
        ref_sr: int = 16000,
        gen_sr: int = 44100,
    ) -> AudioMetricsResult:
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
            reference_audio = librosa.resample(reference_audio.astype(np.float32), orig_sr=ref_sr, target_sr=target_sr)
        if gen_sr != target_sr:
            generated_audio = librosa.resample(generated_audio.astype(np.float32), orig_sr=gen_sr, target_sr=target_sr)

        # Convert to numpy and ensure float32
        reference_audio = reference_audio.astype(np.float32)
        generated_audio = generated_audio.astype(np.float32)

        # Convert to torch tensors for SI-SDR calculation
        reference_tensor = torch.tensor(reference_audio)[None, :].to(self.device,dtype=torch.float32)
        generated_tensor = torch.tensor(generated_audio)[None, :].to(self.device,dtype=torch.float32)

        # Ensure same length
        min_len = min(reference_tensor.shape[1], generated_tensor.shape[1])
        reference_tensor = reference_tensor[:, :min_len]
        generated_tensor = generated_tensor[:, :min_len]

        # Calculate SI-SDR using SQUIM
        try:
            max_audio_length = 15 * target_sr
            model = SQUIM_OBJECTIVE.get_model().to(self.device)
            with torch.no_grad():
                reference_tensor = reference_tensor[:, :min(max_audio_length, reference_tensor.shape[1])]
                generated_tensor = generated_tensor[:, :min(max_audio_length, generated_tensor.shape[1])]
                _, _, sisdr_score = model(generated_tensor)
                sisdr_score = sisdr_score.cpu()[0]
        except Exception as e:
            print(f"SI-SDR calculation failed: {str(e)}")
            sisdr_score = float('-inf')
        finally:
            if 'model' in locals():
                model.cpu()
                del model
                torch.cuda.empty_cache()

        # Resample reference and generated audio to 16kHz for PESQ
        reference_audio_16k = librosa.resample(reference_audio, orig_sr=target_sr, target_sr=16000)
        generated_audio_16k = librosa.resample(generated_audio, orig_sr=target_sr, target_sr=16000)

        # Resample to 24kHz for STOI (as it works better with this sample rate)
        reference_audio_24k = librosa.resample(reference_audio, orig_sr=target_sr, target_sr=24000)
        generated_audio_24k = librosa.resample(generated_audio, orig_sr=target_sr, target_sr=24000)

        # Calculate standard metrics
        try:
            pesq_score = pesq(16000, reference_audio_16k, generated_audio_16k, 'wb')
        except Exception as e:
            print(f"PESQ calculation failed: {str(e)}")
            pesq_score = -1.0

        try:
            stoi_score = stoi(reference_audio_24k[:min(len(reference_audio_24k),len(generated_audio_24k))], generated_audio_24k[:min(len(reference_audio_24k),len(generated_audio_24k))], 24000, extended=False)
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
            sim_o = np.mean([
                np.dot(mfcc_ref[:, i], mfcc_gen[:, i]) /
                (np.linalg.norm(mfcc_ref[:, i]) * np.linalg.norm(mfcc_gen[:, i]) + 1e-8)
                for i in range(min(mfcc_ref.shape[1], mfcc_gen.shape[1]))
            ])
        except Exception as e:
            print(f"SIM-O calculation failed: {str(e)}")
            sim_o = -1.0

        # Calculate SIM-R (Rhythm Similarity)
        try:
            # Using onset strength envelope
            onset_env_ref = librosa.onset.onset_strength(y=reference_audio_24k, sr=24000)
            onset_env_gen = librosa.onset.onset_strength(y=generated_audio_24k, sr=24000)

            # Normalize onset envelopes
            onset_env_ref = (onset_env_ref - np.mean(onset_env_ref)) / (np.std(onset_env_ref) + 1e-8)
            onset_env_gen = (onset_env_gen - np.mean(onset_env_gen)) / (np.std(onset_env_gen) + 1e-8)

            # Calculate correlation coefficient
            sim_r = np.corrcoef(onset_env_ref[:min(len(onset_env_ref), len(onset_env_gen))],
                               onset_env_gen[:min(len(onset_env_ref), len(onset_env_gen))])[0, 1]
        except Exception as e:
            print(f"SIM-R calculation failed: {str(e)}")
            sim_r = -1.0

        return AudioMetricsResult(
            pesq=float(pesq_score),
            stoi=float(stoi_score),
            si_sdr=float(sisdr_score),
            sim_o=float(sim_o),
            sim_r=float(sim_r)
        )

    # def calculate_metrics(
    #     self,
    #     reference_audio: Union[np.ndarray, torch.Tensor],
    #     generated_audio: Union[np.ndarray, torch.Tensor],
    #     sr: int = 24000
    # ) -> AudioMetricsResult:
    #     """Calculate PESQ, STOI, SI-SDR, SIM-O, and SIM-R metrics"""
    #     # Convert to numpy and ensure float32
    #     if isinstance(reference_audio, torch.Tensor):
    #         reference_audio = reference_audio.detach().cpu().numpy()
    #     if isinstance(generated_audio, torch.Tensor):
    #         generated_audio = generated_audio.detach().cpu().numpy()


    #     reference_audio = reference_audio.astype(np.float32)
    #     generated_audio = generated_audio.astype(np.float32)

    #     # Ensure same length
    #     min_len = min(len(reference_audio), len(generated_audio))
    #     reference_audio = reference_audio[:min_len]
    #     generated_audio = generated_audio[:min_len]

    #     # Normalize audio
    #     reference_audio = reference_audio / (np.max(np.abs(reference_audio)) + 1e-8)
    #     generated_audio = generated_audio / (np.max(np.abs(generated_audio)) + 1e-8)

    #     # Resample to 16kHz for PESQ
    #     reference_audio_16k = librosa.resample(reference_audio, orig_sr=sr, target_sr=16000)
    #     generated_audio_16k = librosa.resample(generated_audio, orig_sr=sr, target_sr=16000)

    #     # Calculate standard metrics
    #     try:
    #         pesq_score = pesq(16000, reference_audio_16k, generated_audio_16k, 'wb')
    #     except Exception as e:
    #         print(f"PESQ calculation failed: {str(e)}")
    #         pesq_score = -1.0

    #     try:
    #         stoi_score = stoi(reference_audio_16k, generated_audio_16k, 16000, extended=False)
    #     except Exception as e:
    #         print(f"STOI calculation failed: {str(e)}")
    #         stoi_score = -1.0

    #     try:
    #         sisdr_score = si_sdr(reference_audio_16k.astype(np.float64), generated_audio_16k.astype(np.float64))
    #     except Exception as e:
    #         print(f"SI-SDR calculation failed: {str(e)}")
    #         sisdr_score = float('-inf')

    #     # Calculate SIM-O (Overall Similarity)
    #     try:
    #         # Using MFCCs for overall spectral similarity
    #         mfcc_ref = librosa.feature.mfcc(y=reference_audio, sr=sr, n_mfcc=13)
    #         mfcc_gen = librosa.feature.mfcc(y=generated_audio, sr=sr, n_mfcc=13)

    #         # Normalize MFCCs
    #         mfcc_ref = (mfcc_ref - np.mean(mfcc_ref)) / (np.std(mfcc_ref) + 1e-8)
    #         mfcc_gen = (mfcc_gen - np.mean(mfcc_gen)) / (np.std(mfcc_gen) + 1e-8)

    #         # Calculate cosine similarity for each frame and take mean
    #         sim_o = np.mean([
    #             np.dot(mfcc_ref[:, i], mfcc_gen[:, i]) /
    #             (np.linalg.norm(mfcc_ref[:, i]) * np.linalg.norm(mfcc_gen[:, i]) + 1e-8)
    #             for i in range(min(mfcc_ref.shape[1], mfcc_gen.shape[1]))
    #         ])
    #     except Exception as e:
    #         print(f"SIM-O calculation failed: {str(e)}")
    #         sim_o = -1.0

    #     # Calculate SIM-R (Rhythm Similarity)
    #     try:
    #         # Using onset strength envelope
    #         onset_env_ref = librosa.onset.onset_strength(y=reference_audio, sr=sr)
    #         onset_env_gen = librosa.onset.onset_strength(y=generated_audio, sr=sr)

    #         # Normalize onset envelopes
    #         onset_env_ref = (onset_env_ref - np.mean(onset_env_ref)) / (np.std(onset_env_ref) + 1e-8)
    #         onset_env_gen = (onset_env_gen - np.mean(onset_env_gen)) / (np.std(onset_env_gen) + 1e-8)

    #         # Calculate correlation coefficient
    #         sim_r = np.corrcoef(onset_env_ref[:min(len(onset_env_ref), len(onset_env_gen))],
    #                            onset_env_gen[:min(len(onset_env_ref), len(onset_env_gen))])[0, 1]
    #     except Exception as e:
    #         print(f"SIM-R calculation failed: {str(e)}")
    #         sim_r = -1.0

    #     return AudioMetricsResult(
    #         pesq=float(pesq_score),
    #         stoi=float(stoi_score),
    #         si_sdr=float(sisdr_score),
    #         sim_o=float(sim_o),
    #         sim_r=float(sim_r)
    #     )

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
        for idx, sample in tqdm(enumerate(samples), total=len(samples), desc="Processing samples"):
            try:
                if isinstance(sample, tuple):
                    text, reference_audio = sample
                    print(text,prompt)
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
                print({'PESQ': metrics.pesq,
                    'STOI': metrics.stoi,
                    'SI-SDR': metrics.si_sdr,
                    'SIM-O': metrics.sim_o,
                    'SIM-R': metrics.sim_r})

                result_dict = {
                    'PESQ': metrics.pesq,
                    'STOI': metrics.stoi,
                    'SI-SDR': metrics.si_sdr,
                    'SIM-O': metrics.sim_o,
                    'SIM-R': metrics.sim_r,
                    'sample_idx': idx
                }

                if text:
                    result_dict['text'] = text
                if batch_metadata:
                    result_dict.update(batch_metadata)

                if save_audio:
                    audio_filename = f"gen_{idx}.wav"
                    audio_filename_ = f"orig_{idx}.wav"
                    audio_path = os.path.join(self.gen_audio_dir, audio_filename)
                    audio_path_ = os.path.join(self.gen_audio_dir, audio_filename_)
                    sf.write(audio_path, generated_audio.astype(np.float32), 44100)
                    sf.write(audio_path_, reference_audio.astype(np.float32), 16000)
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
        save_audio=True,
    ):
        """Evaluate metrics on LibriSpeech samples"""
        print(f"Loading LibriSpeech dataset ({subset})...")
        dataset = torchaudio.datasets.LIBRISPEECH("./data", url=subset, download=True)

        # Randomly sample entries
        indices = range(600,801)#random.sample(range(len(dataset)), num_samples)

        samples = []
        metadata = []
        print("Preparing samples...")
        for idx in tqdm(indices, desc="Loading audio files"):
            waveform, sample_rate, text, speaker_id, chapter_id, utterance_id = dataset[torch.tensor(idx).long()]
            reference_audio = waveform.numpy().squeeze()
            # if sample_rate != 24000:
            #     reference_audio = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=24000)
            samples.append((text, reference_audio))
            metadata.append({
                'speaker_id': speaker_id,
                'chapter_id': chapter_id,
                'utterance_id': utterance_id
            })

        print("Running evaluation...")
        results_df = self.evaluate_batch(
            samples,
            prompt=prompt,
            batch_metadata={'dataset': 'librispeech', 'subset': subset},
            save_audio=save_audio
        )

        # Add metadata
        for idx, meta in enumerate(metadata):
            for key, value in meta.items():
                results_df.loc[results_df['sample_idx'] == idx, key] = value

        # Calculate and print average metrics
        avg_metrics = results_df[['PESQ', 'STOI', 'SI-SDR', 'SIM-O', 'SIM-R']].mean()
        std_metrics = results_df[['PESQ', 'STOI', 'SI-SDR', 'SIM-O', 'SIM-R']].std()

        print("\nMetrics Summary:")
        for metric in avg_metrics.index:
            print(f"{metric}: Mean = {avg_metrics[metric]:.4f}, Std = {std_metrics[metric]:.4f}")

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

