from abc import ABC, abstractmethod
from typing import Optional

import torch
import torchaudio

from audiotools import AudioSignal

from BigCodec.vq import CodecEncoder
from BigCodec.vq.codec_decoder import CodecDecoder

from speechtokenizer import SpeechTokenizer

from WavTokenizer.decoder.pretrained import WavTokenizer


class AudioTokenizer(ABC):
    def __init__(
        self,
        sampling_rate_asr,
        sampling_rate_tts,
        n_original_tokens,
        start_audio_token_id: Optional[int] = None,
        end_audio_token_id: Optional[int] = None,
        start_sequence_token_id: Optional[int] = None,
        device="cuda",
    ):
        self.sampling_rate_asr = sampling_rate_asr
        self.sampling_rate_tts = sampling_rate_tts

        self.n_original_tokens = n_original_tokens
        self.start_audio_token_id = start_audio_token_id
        self.end_audio_token_id = end_audio_token_id
        self.start_sequence_token_id = start_sequence_token_id
        self.device = device

    def resample(self, audio, sr):
        if sr != self.sampling_rate_asr:
            return torchaudio.functional.resample(audio, sr, self.sampling_rate_asr)
        return audio

    def get_audio_start_end_tokens(
        self,
        tokens: torch.Tensor,
    ):
        # find start index of audio tokens
        if self.start_audio_token_id is not None:
            start = torch.nonzero(tokens == self.start_audio_token_id)
            start = start[0, -1] + 1 if len(start) else 0
        else:
            start = 0

        # find end index of audio tokens
        if self.end_audio_token_id is not None:
            end = torch.nonzero(tokens == self.end_audio_token_id)
            end = end[0, -1] if len(end) else tokens.shape[-1]
        else:
            end = tokens.shape[-1]

        assert start < end, (
            f"Start of audio must be before end. Found: start - {start}, end - {end}"
        )

        return start, end

    @abstractmethod
    def _encode(self, audio):
        pass

    def encode(self, audio):
        codes = self._encode(audio)
        raw_tokens = codes + self.n_original_tokens
        audio_tokens = raw_tokens.view(1, -1)
        tokens = torch.cat(
            [
                self.start_sequence_token_id,
                self.start_audio_token_id,
                audio_tokens,
                self.end_audio_token_id,
            ],
            dim=1,
        )
        return tokens

    @abstractmethod
    def decode(self, tokens):
        pass


class BigCodecsAudioTokenizer(AudioTokenizer):
    def __init__(self, checkpoint_path: str = "checkpoints/BigCodecs.pt", **kwargs):
        sr = 16000
        super().__init__(sr, sr, **kwargs)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        encoder = CodecEncoder()
        encoder.load_state_dict(ckpt["CodecEnc"])
        self.encoder = encoder.eval().cuda()

        decoder = CodecDecoder()
        decoder.load_state_dict(ckpt["generator"])
        self.decoder = decoder.eval().cuda()

    def _encode(self, audio):
        wav, sr = audio["array"], audio["sampling_rate"]

        if wav.shape(0) > 1:
            wav = wav[:1, :]

        wav = self.resample(wav, sr)
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            vq_emb = self.encoder(wav.unsqueeze(1))
            _, vq_code, _ = self.decoder(vq_emb, vq=True)
            codes = vq_code.squeeze(0, 1)

        return codes

    def decode(self, tokens):
        # find audio start and end tokens
        start, end = self.get_audio_start_end_tokens(
            tokens,
        )

        audio_tokens = tokens[start:end] % self.n_original_tokens
        audio_tokens = audio_tokens.reshape(1, -1, 1)
        emb = self.decoder.vq2emb(audio_tokens).transpose(1, 2)
        audio = self.decoder(emb, vq=False).squeeze().detach().cpu().numpy()

        return AudioSignal(audio, self.sampling_rate_tts)


class SpeechAudioTokenizer(AudioTokenizer):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/SpeechTokenizer.pt",
        config_path: str = "checkpoints/config.json",
        n_channels_asr=1,
        n_channels_tts=3,
        **kwargs,
    ):
        quantizer = SpeechTokenizer.load_from_checkpoint(config_path, checkpoint_path)

        sr = quantizer.sample_rate

        super().__init__(sr, sr, **kwargs)

        self.quantizer = quantizer
        self.quantizer.to(self.device)

        self.n_channels_asr = n_channels_asr
        self.n_channels_tts = n_channels_tts

    def _encode(self, audio):
        wav, sr = audio["array"], audio["sampling_rate"]
        wav = torch.from_numpy(wav).to(self.device)
        wav = wav.unsqueeze(0)

        if wav.shape[0] > 1:
            wav = wav[:1]

        wav = self.resample(wav, sr)
        wav = wav.unsqueeze(0)

        with torch.no_grad():
            codes = self.quantizer.encode(wav)
            codes = codes.squeeze(1)

        return codes[: self.n_channels_asr, :]

    def decode(self, tokens):
        # find audio start and end tokens
        start, end = self.get_audio_start_end_tokens(tokens)

        audio_tokens = tokens[start:end] % self.n_original_tokens
        remainder = audio_tokens.shape[-1] % self.n_channels_tts

        if remainder:
            # pad if last frame is incomplete
            # zero padding is used now, for speechtokenizer using get_audio_padding_tokens is also possible
            pad_tokens = torch.zeros(
                1, self.n_channels_tts - remainder, device="cuda", dtype=torch.long
            )
            audio_tokens = torch.cat([audio_tokens, pad_tokens], dim=0)

        transposed = audio_tokens.view(-1, self.n_channels_tts).t()
        codes = transposed.view(self.n_channels_tts, 1, -1).to(self.device)

        audio = self.quantizer.decode(codes).squeeze(0)
        return AudioSignal(audio.detach().cpu().numpy(), self.sampling_rate_tts)


class WavAudioTokenizer(AudioTokenizer):
    def __init__(
        self,
        checkpoint_path: str = "wavtokenizer_large_unify_600_24k.ckpt",
        config_path: str = "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        **kwargs,
    ):
        quantizer = WavTokenizer.from_pretrained0802(config_path, checkpoint_path)
        quantizer = quantizer.to(self.device)

        sr = quantizer.feature_extarctor.encodec.sample_rate

        super().__init__(sr, sr, **kwargs)

        self.quantizer = quantizer

    def _encode(self, audio):
        wav, sr = audio["array"], audio["sampling_rate"]

        if wav.shape(0) > 1:
            wav = wav[:1, :]

        wav = self.resample(wav, sr)
        bandwidth_id = torch.tensor([0])

        audio = wav.to(self.device)
        _, codes = self.quantizer.encode_infer(audio, bandwidth_id=bandwidth_id)
        codes = codes.squeeze(1)

        return codes

    def decode(self, tokens):
        # find audio start and end tokens
        start, end = self.get_audio_start_end_tokens(
            tokens,
        )

        # subtract length of original vocabulary -> tokens in range [0, n_codebooks)
        audio_tokens = tokens[start:end] % self.n_original_tokens

        transposed = audio_tokens.view(-1, 1).t()
        codes = transposed.view(1, 1, -1).to(self.device)

        features = self.quantizer.codes_to_features(codes)
        bandwidth_id = torch.tensor([0], device=self.device)

        audio = self.quantizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)

        return AudioSignal(audio.detach().cpu().numpy(), self.sampling_rate_tts)


class MixedAudioTokenizer(AudioTokenizer):
    def __init__(
        self, asr_tokenizer: AudioTokenizer, tts_tokenizer: AudioTokenizer, **kwargs
    ):
        super().__init__(
            asr_tokenizer.sampling_rate_asr, tts_tokenizer.sampling_rate_tts, **kwargs
        )
        self.asr_tokenizer = asr_tokenizer
        self.tts_tokenizer = tts_tokenizer

    def _encode(self, audio_path):
        return self.asr_tokenizer.encode(audio_path)

    def encode(self, audio_path):
        return self._encode(audio_path)

    def decode(self, tokens):
        return self.tts_tokenizer.decode(tokens)
