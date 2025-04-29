import os
import subprocess
import tempfile
import hashlib

from typing import Optional, Tuple, Dict

import numpy as np
import soundfile as sf

from evaluate_tts import TTSEvaluator, evaluate_on_mozilla


class FishTTSWrapper:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/fish-speech-1.5",
        device: str = "cuda:0",
        use_compile: bool = True,
        use_half: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Fish TTS wrapper.

        Args:
            checkpoint_path: Path to Fish Speech checkpoints directory
            device: Device to run inference on
            use_compile: Whether to use compiled CUDA kernels for faster inference
            use_half: Whether to use half precision (for GPUs without bf16 support)
            cache_dir: Directory to cache voice prompts (defaults to temp directory)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_compile = use_compile
        self.use_half = use_half

        # Paths to specific model files
        self.vqgan_path = os.path.join(
            checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        )

        # Set up caching
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        os.makedirs(self.cache_dir, exist_ok=True)

        # Cache for voice prompts
        self.prompt_cache: Dict[str, str] = {}

    def _run_command(self, command: list) -> subprocess.CompletedProcess:
        """Run a command and handle errors."""
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e.stderr}")
            raise

    def _get_audio_hash(self, audio_path: str) -> str:
        """Generate a hash for the audio file."""
        with open(audio_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def generate_voice_prompt(self, reference_audio: str) -> str:
        """Generate voice prompt from reference audio.

        Args:
            reference_audio: Path to reference audio file

        Returns:
            Path to generated prompt file
        """
        # Check cache first
        audio_hash = self._get_audio_hash(reference_audio)
        if audio_hash in self.prompt_cache:
            return self.prompt_cache[audio_hash]

        # Generate new prompt
        prompt_path = os.path.join(self.cache_dir, f"prompt_{audio_hash}.npy")

        # Only generate if not already exists
        if not os.path.exists(prompt_path):
            command = [
                "/workspace/.venv/bin/python",
                "tools/vqgan/inference.py",
                "-i",
                reference_audio,
                "--checkpoint-path",
                self.vqgan_path,
                "-o",
                prompt_path,  # Specify output path
            ]
            self._run_command(command)

        # Cache the result
        self.prompt_cache[audio_hash] = prompt_path
        return prompt_path

    def generate_semantic_tokens(
        self,
        text: str,
        reference_text: Optional[str] = None,
        prompt_tokens_path: Optional[str] = None,
        num_samples: int = 1,
    ) -> str:
        """Generate semantic tokens from text.

        Args:
            text: Text to convert to speech
            reference_text: Reference text for voice matching
            prompt_tokens_path: Path to voice prompt tokens (optional)
            num_samples: Number of samples to generate

        Returns:
            Path to generated semantic tokens file
        """
        # Create unique output path based on inputs
        inputs_hash = hashlib.md5(
            f"{text}{reference_text}{prompt_tokens_path}".encode()
        ).hexdigest()
        output_path = os.path.join(self.cache_dir, f"codes_{inputs_hash}.npy")
        text = text.capitalize()

        if not os.path.exists(output_path):
            command = [
                "python",
                "fish-speech/fish_speech/models/text2semantic/inference.py",
                "--text",
                text,
                "--prompt-text",
                reference_text,
                "--checkpoint-path",
                self.checkpoint_path,
                "--num-samples",
                "1",
                # "--output-path", output_path
            ]

            if prompt_tokens_path:
                command.extend(["--prompt-tokens", prompt_tokens_path])

            if self.use_compile:
                command.append("--compile")

            if self.use_half:
                command.append("--half")

            self._run_command(command)

        return output_path

    def generate_audio(self, semantic_tokens_path: str) -> str:
        """Generate audio from semantic tokens.

        Args:
            semantic_tokens_path: Path to semantic tokens file

        Returns:
            Path to generated audio file
        """
        # Create unique output path
        tokens_hash = self._get_audio_hash("codes_0.npy")
        output_path = os.path.join(self.cache_dir, f"audio_{tokens_hash}.wav")

        if not os.path.exists(output_path):
            command = [
                "/workspace/.venv/bin/python",
                "tools/vqgan/inference.py",
                "-i",
                "/workspace/aulate/codes_0.npy",
                "--checkpoint-path",
                self.vqgan_path,
                "-o",
                output_path,
            ]
            self._run_command(command)

        return output_path

    def infer_text_to_audio(
        self,
        text: str,
        reference_text: Optional[str] = None,
        reference_audio: Optional[str] = None,
    ) -> Tuple[int, np.ndarray]:
        """Generate audio from text.

        Args:
            text: Text to convert to speech
            reference_text: Reference text for voice matching
            reference_audio: Optional path to reference audio for voice cloning

        Returns:
            Tuple of (sampling_rate, audio_waveform)
        """
        # Generate voice prompt if reference audio provided
        prompt_path = None
        if reference_audio:
            prompt_path = self.generate_voice_prompt(reference_audio)

        # Generate semantic tokens
        semantic_tokens_path = self.generate_semantic_tokens(
            text, reference_text, prompt_path
        )

        # Generate audio
        audio_path = self.generate_audio(semantic_tokens_path)

        # Load and return audio
        audio, sr = sf.read(audio_path)
        return sr, audio

    def cleanup(self):
        """Cleanup temporary files."""
        if os.path.exists(self.cache_dir):
            import shutil

            shutil.rmtree(self.cache_dir)

    def __del__(self):
        """Cleanup on object destruction if cache_dir was temporary."""
        if self.cache_dir.startswith(tempfile.gettempdir()):
            self.cleanup()


if __name__ == "__main__":
    wrapper = FishTTSWrapper()

    evaluator = TTSEvaluator(inference_fn=wrapper.infer_text_to_audio)

    results_df = evaluate_on_mozilla(evaluator, num_samples=50, prompt=None)
