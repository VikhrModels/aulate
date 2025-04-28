import os

import torch

import torchaudio
from ichigo.asr import transcribe

from evaluate_asr import evaluate_on_librispeech, ASREvaluator


class LlamaIchigoWrapper:
    def infer_audio_to_text(
        self,
        audio,
        **kwargs,
    ):
        arr = torch.tensor([audio["array"]])
        torchaudio.save("temp.wav", arr, audio["sampling_rate"])
        text = transcribe("temp.wav")
        os.remove("temp.wav")
        return text


if __name__ == "__main__":
    custom_evaluator = LlamaIchigoWrapper()

    evaluator = ASREvaluator(
        inference_fn=custom_evaluator.infer_audio_to_text,
    )

    results_df = evaluate_on_librispeech(
        evaluator=evaluator,
    )
