import sys
sys.path.append("WavTokenizer")
from aulate.metrics_evaluation import AudioMetricsEvaluator
import sys
import os
import argparse
from tqdm import tqdm

def evaluate_checkpoints(checkpoint_parent, tokenizer_config, tokenizer_checkpoint,
                         num_samples, prompt, output_dir):
    # Получаем список директорий чекпоинтов
    checkpoint_dirs = sorted([
        d for d in os.listdir(checkpoint_parent)
        if os.path.isdir(os.path.join(checkpoint_parent, d))
    ])

    # Убедимся, что папка для сохранения результатов существует
    os.makedirs(output_dir, exist_ok=True)

    # Словарь для хранения результатов по каждому чекпоинту (необязательно)
    results_dfs = {}

    # Проходим по каждому чекпоинту с использованием tqdm
    for checkpoint in tqdm(checkpoint_dirs, desc="Evaluating checkpoints"):
        base_model_path = os.path.join(checkpoint_parent, checkpoint)
        evaluator = AudioMetricsEvaluator(
            base_model=base_model_path,
            speechtokenizer_config=tokenizer_config,
            speechtokenizer_checkpoint=tokenizer_checkpoint
        )
        try:
            results_df = evaluator.evaluate_on_librispeech(
                num_samples=num_samples,
                prompt=prompt,
                save_audio=False
            )
            results_dfs[checkpoint] = results_df

            # Сохраняем таблицу с результатами в CSV файл с именем чекпоинта
            csv_filename = os.path.join(output_dir, f"results_{checkpoint}.csv")
            results_df.to_csv(csv_filename, index=False)
            print(f"Saved results for {checkpoint} to {csv_filename}")
        except Exception as e:
            print(f"Error processing {base_model_path}: {e}")

    return results_dfs

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a given folder using AudioMetricsEvaluator.")
    parser.add_argument("--checkpoint_parent", type=str, required=True,
                        help="Path to the parent folder containing checkpoint directories.")
    parser.add_argument("--speechtokenizer_config", type=str, required=True,
                        help="Path to the speech tokenizer configuration file.")
    parser.add_argument("--speechtokenizer_checkpoint", type=str, required=True,
                        help="Path to the speech tokenizer checkpoint file.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate on (default: 100).")
    parser.add_argument("--prompt", type=str, default=("with a male speaker delivers a very monotone and high-pitched speech with a very fast speed "
                                                        "in a setting with almost no noise, creating a clear and loud recording."),
                        help="Evaluation prompt text.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the resulting CSV files (default: 'results').")
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate_checkpoints(
        checkpoint_parent=args.checkpoint_parent,
        tokenizer_config=args.speechtokenizer_config,
        tokenizer_checkpoint=args.speechtokenizer_checkpoint,
        num_samples=args.num_samples,
        prompt=args.prompt,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
