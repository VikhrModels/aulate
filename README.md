# Audio Metrics Evaluation Library

A flexible library for evaluating audio generation models using standard metrics (PESQ, STOI, SI-SDR) and similarity metrics (SIM-O, SIM-R). Designed to work with any text-to-speech or audio generation model.

## Features

- Calculate standard audio quality metrics:
  - PESQ (Perceptual Evaluation of Speech Quality)
  - STOI (Short-Time Objective Intelligibility)
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- Calculate similarity metrics:
  - SIM-O (Overall Spectral Similarity)
  - SIM-R (Rhythm Similarity)
- Support for both default and custom model implementations
- Batch evaluation capabilities
- Built-in LibriSpeech evaluation
- Flexible audio saving and metadata handling

## Installation

```bash
pip install pesq pystoi torch-mir-eval torchaudio pandas torch transformers soundfile tqdm numpy librosa speechtokenizer beartype
pip install git+https://github.com/descriptinc/audiotools
git clone https://github.com/jishengpeng/WavTokenizer.git
mkdir data
```

## Usage

### Basic Usage with Default Implementation

```python
from aulate.metrics_evaluation import AudioMetricsEvaluator

# Initialize with default implementation
evaluator = AudioMetricsEvaluator(
    base_model='path/to/model',
    speechtokenizer_config="path/to/config.yaml",
    speechtokenizer_checkpoint="path/to/checkpoint.ckpt"
)

# Evaluate on LibriSpeech
results = evaluator.evaluate_on_librispeech(
    num_samples=10,
    prompt="with a natural speaking voice",
    save_audio=True
)
```

# OR run in cli

```python
python test.py --checkpoint_parent "../results/Qwen2.5-0.5B_tts_wav_1" --speechtokenizer_config "WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml" --speechtokenizer_checkpoint "audiotokenizer/wavtokenizer_large_unify_600_24k.ckpt" --num_samples 100 --prompt "your prompt here" --output_dir "results"
```

### Using Custom Implementation

```python
def custom_decode(tokens):
    # Your custom decode implementation
    pass

def custom_inference(text, prompt):
    # Your custom inference implementation
    pass

# Initialize with custom functions
evaluator = AudioMetricsEvaluator(
    decode_fn=custom_decode,
    inference_fn=custom_inference
)
```

### Mixing Default and Custom Implementations

```python
# Start with default implementation
evaluator = AudioMetricsEvaluator(
    base_model='path/to/model',
    speechtokenizer_config="path/to/config.yaml",
    speechtokenizer_checkpoint="path/to/checkpoint.ckpt"
)

# Later override with custom implementation
evaluator.set_decode_fn(custom_decode)
evaluator.set_inference_fn(custom_inference)
```

### Batch Evaluation

```python
# Prepare samples
samples = [
    ("Hello world", reference_audio1),  # For text-to-audio
    ("Test sentence", reference_audio2)
]

# Evaluate batch
results = evaluator.evaluate_batch(
    samples,
    prompt="with a natural voice",
    save_audio=True,
    batch_metadata={'experiment': 'test1'}
)
```

## API Reference

### AudioMetricsEvaluator

Main class for audio metrics evaluation.

#### Parameters

- `base_model` (Optional[str]): Path to the base model
- `speechtokenizer_config` (Optional[str]): Path to speechtokenizer config
- `speechtokenizer_checkpoint` (Optional[str]): Path to speechtokenizer checkpoint
- `decode_fn` (Optional[Callable]): Custom function to decode audio tokens
- `inference_fn` (Optional[Callable]): Custom function for text-to-audio generation
- `device` (str): Device to run computations on (default: "cuda")
- `cache_dir` (str): Directory for model caching (default: ".")
- `results_dir` (str): Directory to save evaluation results (default: "evaluation_results")

#### Methods

- `evaluate_batch(samples, prompt=None, batch_metadata=None, save_audio=False)`: Evaluate metrics on a batch of samples
- `evaluate_on_librispeech(num_samples=200, prompt="...", subset="test-clean")`: Evaluate on LibriSpeech dataset
- `set_decode_fn(decode_fn)`: Set custom decode function
- `set_inference_fn(inference_fn)`: Set custom inference function

## Output Format

The evaluation results are returned as a pandas DataFrame with the following columns:

- PESQ: Perceptual Evaluation of Speech Quality score (0-4.5)
- STOI: Short-Time Objective Intelligibility score (0-1)
- SI-SDR: Scale-Invariant Signal-to-Distortion Ratio (dB)
- SIM-O: Overall Spectral Similarity score (0-1)
- SIM-R: Rhythm Similarity score (-1 to 1)
- sample_idx: Index of the sample
- text: Input text (for text-to-audio samples)
- audio_path: Path to saved audio file (if save_audio=True)
- Additional metadata columns as provided

### Metrics Description

- **PESQ**: Industry standard for speech quality assessment. Higher is better (range: 0-4.5).
- **STOI**: Measures speech intelligibility. Higher is better (range: 0-1).
- **SI-SDR**: Measures signal distortion. Higher is better (typical range: -∞ to +∞ dB).
- **SIM-O**: Measures overall spectral similarity using MFCCs. Higher means more similar (range: 0-1).
- **SIM-R**: Measures rhythm similarity using onset patterns. Higher means more similar rhythm (range: -1 to 1).

Results are automatically saved to CSV files in the specified results directory.

## License

MIT License 
