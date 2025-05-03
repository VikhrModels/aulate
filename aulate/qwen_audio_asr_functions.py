import argparse
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from evaluate_asr import evaluate_on_librispeech, ASREvaluator


class QwenAudioWrapper:
    def __init__(self, model_path: str = "Qwen/Qwen2-Audio-7B"):
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    def infer_audio_to_text(
        self,
        audio,
        **kwargs,
    ):
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the audio:"
        inputs = self.processor(text=prompt, audios=audio["array"], return_tensors="pt")
        generated_ids = self.model.generate(**inputs, max_length=256)
        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR model on LibriSpeech")
    parser.add_argument("--model_path", type=str, help="Model name")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate on"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed to use."
    )

    args = parser.parse_args()

    custom_evaluator = QwenAudioWrapper(args.model_path)

    evaluator = ASREvaluator(
        inference_fn=custom_evaluator.infer_audio_to_text,
    )

    results_df = evaluate_on_librispeech(
        evaluator=evaluator,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
    )
