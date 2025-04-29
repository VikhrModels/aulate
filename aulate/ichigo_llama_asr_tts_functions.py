from ichigo.asr import transcribe, IchigoASR

from evaluate_asr import evaluate_on_librispeech, ASREvaluator


class LlamaIchigoWrapper:
    def __init__(self, config="ichigo-asr-2501-en-vi"):
        self.model = IchigoASR(config=config)

    def infer_audio_to_text(
        self,
        audio,
        **kwargs,
    ):
        text = transcribe(audio["audio_path"], None, (".wav",))
        return text[0]


if __name__ == "__main__":
    custom_evaluator = LlamaIchigoWrapper()

    evaluator = ASREvaluator(
        inference_fn=custom_evaluator.infer_audio_to_text,
    )

    results_df = evaluate_on_librispeech(evaluator=evaluator, save_audio=True)
