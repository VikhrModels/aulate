import argparse
from ichigo.asr import transcribe, IchigoASR

from evaluate_asr import evaluate_on_librispeech, ASREvaluator


class LlamaIchigoWrapper:
    def __init__(self, config="ichigo-asr-2501-en-vi"):
        print(config)
        self.model = IchigoASR(config=config)

    def infer_audio_to_text(
        self,
        audio,
        **kwargs,
    ):
        text = transcribe(audio["audio_path"], None, (".wav",))
        return text[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR model on LibriSpeech")
    parser.add_argument("--config", type=str, help="Model name")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate on"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed to use."
    )

    args = parser.parse_args()

    custom_evaluator = LlamaIchigoWrapper(args.config)

    evaluator = ASREvaluator(
        inference_fn=custom_evaluator.infer_audio_to_text,
    )

    results_df = evaluate_on_librispeech(
        evaluator=evaluator,
        save_audio=True,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
    )
