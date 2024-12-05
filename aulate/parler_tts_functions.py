import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import numpy as np
from typing import Optional, Union, Tuple

class ParlerTTSWrapper:
    def __init__(
        self,
        model_name: str = "parler-tts/parler-tts-mini-v1",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize Parler TTS model and tokenizer."""
        self.device = device
        self.dtype = dtype

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
        ).to(device, dtype=dtype)

        # Get model config values
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        self.frame_rate = self.model.audio_encoder.config.frame_rate

    def decode_audio(
        self,
        tokens: Union[torch.Tensor, np.ndarray],
        prompt: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Decode audio from tokens.

        Args:
            tokens: Input tokens to decode
            prompt: Optional prompt tokens

        Returns:
            Audio waveform as numpy array
        """
        # Convert to tensor if needed
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens)

        tokens = tokens.to(self.device)
        if prompt is not None:
            prompt = prompt.to(self.device)

        # Generate audio without streaming
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=tokens.unsqueeze(0),
                prompt_input_ids=prompt.unsqueeze(0) if prompt is not None else None,
                do_sample=True,
                temperature=1.0,
                min_new_tokens=10,
            )

        # Convert to numpy array
        audio = outputs.cpu().numpy().squeeze()
        return audio

    def infer_text_to_audio(
        self,
        text: str,
        description: str = "Speaking naturally.",
        temperature: float = 1.0,
        min_new_tokens: int = 10,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate audio from text.

        Args:
            text: Input text to synthesize
            description: Voice description/prompt
            temperature: Sampling temperature
            min_new_tokens: Minimum number of tokens to generate

        Returns:
            Tuple of (sampling_rate, audio_waveform)
        """
        # Tokenize inputs
        inputs = self.tokenizer(description, return_tensors="pt").to(self.device)
        text = text.lower()
        fc = text[0].upper()
        text = fc + text[1:]
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate audio without streaming
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                prompt_input_ids=prompt.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_attention_mask=prompt.attention_mask,
                do_sample=True,
                temperature=temperature,
                min_new_tokens=min_new_tokens,
            )

        # Convert to numpy array
        audio = outputs.cpu().numpy().squeeze()
        return self.sampling_rate, audio

# Example usage with metrics evaluator:
"""
from aulate.metrics_evaluation import AudioMetricsEvaluator
from aulate.parler_tts_functions import ParlerTTSWrapper

# Initialize Parler TTS
parler = ParlerTTSWrapper()

# Create custom functions for metrics evaluator
def custom_decode(tokens):
    return parler.decode_audio(tokens)

def custom_inference(text, prompt):
    sr, audio = parler.infer_text_to_audio(text, description=prompt)
    return audio

# Initialize evaluator with custom functions
evaluator = AudioMetricsEvaluator(
    decode_fn=custom_decode,
    inference_fn=custom_inference
)

# Now you can use the evaluator as normal
results = evaluator.evaluate_batch(samples, prompt="Speaking naturally.")
"""