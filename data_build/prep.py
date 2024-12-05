import polars as pl
from datasets import load_dataset, get_dataset_split_names
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_conversation(messages: List[Dict[str, str]], source: str, metadata: Dict = None) -> Dict[str, Any]:
    """Create a standardized conversation entry."""
    return {
        "conversations": messages,
        "source": source,
        "metadata": metadata or {}
    }

def safe_str(value: Any) -> str:
    """Safely convert value to string."""
    return str(value) if value is not None else ""

def process_dataset(df: pd.DataFrame, processor_func) -> pl.DataFrame:
    """Process a dataset using the provided function and convert to polars."""
    conversations = []
    for _, row in df.iterrows():
        try:
            conv = processor_func(row)
            conversations.append(conv)
        except Exception as e:
            logger.warning(f"Error processing row: {e}")
            continue
    return pl.DataFrame(conversations)

def get_first_split(dataset_name: str, config_name: Optional[str] = None) -> str:
    """Get the first available split name for a dataset."""
    try:
        splits = get_dataset_split_names(dataset_name, config_name)
        return splits[0] if splits else "train"
    except Exception as e:
        logger.warning(f"Error getting splits for {dataset_name}: {e}")
        return "train"

def load_awesome_chatgpt_prompts() -> pl.DataFrame:
    """Load and process awesome-chatgpt-prompts dataset."""
    dataset = load_dataset("fka/awesome-chatgpt-prompts")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "system", "content": safe_str(row["prompt"])},
                {"role": "assistant", "content": f"I will act as {safe_str(row['act'])}"}
            ],
            source="fka/awesome-chatgpt-prompts",
            metadata={"category": "system_prompt", "act": safe_str(row["act"])}
        )

    return process_dataset(df, process_row)

def load_roleplay_dataset() -> pl.DataFrame:
    """Load and process roleplay dataset."""
    dataset = load_dataset("hieunguyenminh/roleplay")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "system", "content": safe_str(row.get("system"))},
                {"role": "user", "content": safe_str(row.get("human"))}
            ],
            source="hieunguyenminh/roleplay",
            metadata={"category": "roleplay"}
        )

    return process_dataset(df, process_row)

def load_herman_json() -> pl.DataFrame:
    """Load and process Herman JSON dataset."""
    dataset = load_dataset("SulthanAbiyyu/herman-json-mode")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": safe_str(row.get("instruction"))},
                {"role": "assistant", "content": safe_str(row.get("output"))}
            ],
            source="SulthanAbiyyu/herman-json-mode",
            metadata={"category": "structured_output"}
        )

    return process_dataset(df, process_row)

def load_jailbreak_behaviors() -> pl.DataFrame:
    """Load and process Jailbreak Behaviors dataset."""
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    split = get_first_split("JailbreakBench/JBB-Behaviors", "behaviors")
    df = dataset[split].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": safe_str(row.get("prompt"))}
            ],
            source="JailbreakBench/JBB-Behaviors",
            metadata={
                "category": "jailbreak",
                "target_behavior": safe_str(row.get("target_behavior")),
                "behavior_category": safe_str(row.get("category"))
            }
        )

    return process_dataset(df, process_row)

def load_social_reasoning() -> pl.DataFrame:
    """Load and process Social Reasoning RLHF dataset."""
    dataset = load_dataset("ProlificAI/social-reasoning-rlhf")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": safe_str(row.get("prompt"))},
                {"role": "assistant", "content": safe_str(row.get("chosen"))}
            ],
            source="ProlificAI/social-reasoning-rlhf",
            metadata={
                "category": "social_reasoning",
                "rejected_response": safe_str(row.get("rejected"))
            }
        )

    return process_dataset(df, process_row)

def load_hallucination_dataset() -> pl.DataFrame:
    """Load and process C0uchP0tat0/llm_hallucinations dataset."""
    dataset = load_dataset("C0uchP0tat0/llm_hallucinations")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": f"{safe_str(row.get('summary'))}\n\nQuestion: {safe_str(row.get('question'))}"},
                {"role": "assistant", "content": safe_str(row.get("answer"))}
            ],
            source="C0uchP0tat0/llm_hallucinations",
            metadata={
                "category": "hallucination_test",
                "line_id": int(row.get("line_id", 0)),
                "is_hallucination": bool(row.get("is_hallucination", False))
            }
        )

    return process_dataset(df, process_row)

def load_logic_puzzles() -> pl.DataFrame:
    """Load and process LOGIC-701 dataset."""
    # Process both English and Russian splits
    dfs = []
    for lang in ["en", "ru"]:
        dataset = load_dataset("hivaze/LOGIC-701", lang)
        split = get_first_split("hivaze/LOGIC-701", lang)
        df = dataset[split].to_pandas()
        df["lang"] = lang  # Add language info
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": safe_str(row.get("question"))},
                {"role": "assistant", "content": f"Answer: {safe_str(row.get('answer'))}\n\nExplanation: {safe_str(row.get('explanation'))}"}
            ],
            source="hivaze/LOGIC-701",
            metadata={
                "category": "logic_puzzle",
                "language": safe_str(row.get("lang"))
            }
        )

    return process_dataset(df, process_row)

def load_brain_teasers() -> pl.DataFrame:
    """Load and process brain-teasers dataset."""
    dataset = load_dataset("ErfanMoosaviMonazzah/brain-teasers")
    # Combine both splits
    dfs = []
    for split in ["sp", "wp"]:  # Use both available splits
        df = dataset[split].to_pandas()
        df["split_type"] = split
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": safe_str(row.get("question"))},
                {"role": "assistant", "content": safe_str(row.get("answer"))}
            ],
            source="ErfanMoosaviMonazzah/brain-teasers",
            metadata={
                "category": "brain_teaser",
                "difficulty": safe_str(row.get("difficulty")),
                "split_type": safe_str(row.get("split_type"))
            }
        )

    return process_dataset(df, process_row)

def load_propositional_logic() -> pl.DataFrame:
    """Load and process propositional-logic dataset."""
    dataset = load_dataset("ergotts/propositional-logic")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": f"{safe_str(row.get('premise'))}\n\n{safe_str(row.get('hypothesis'))}"},
                {"role": "assistant", "content": f"{safe_str(row.get('explanation'))}"}
            ],
            source="ergotts/propositional-logic",
            metadata={
                "category": "propositional_logic",
                "label": bool(row.get("label", False))
            }
        )

    return process_dataset(df, process_row)

def load_numina_math() -> pl.DataFrame:
    """Load and process NuminaMath-CoT dataset."""
    dataset = load_dataset("AI-MO/NuminaMath-CoT")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": safe_str(row.get("question"))},
                {"role": "assistant", "content": f"{safe_str(row.get('solution'))}\n\n{safe_str(row.get('answer'))}"}
            ],
            source="AI-MO/NuminaMath-CoT",
            metadata={
                "category": "math",
                "difficulty": safe_str(row.get("difficulty")),
                "topic": safe_str(row.get("topic"))
            }
        )

    return process_dataset(df, process_row)

def load_openmath_instruct() -> pl.DataFrame:
    """Load and process OpenMathInstruct-2 dataset."""
    dataset = load_dataset("nvidia/OpenMathInstruct-2")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": safe_str(row.get("instruction"))},
                {"role": "assistant", "content": safe_str(row.get("response"))}
            ],
            source="nvidia/OpenMathInstruct-2",
            metadata={
                "category": "math",
                "math_category": safe_str(row.get("category"))
            }
        )

    return process_dataset(df, process_row)

def load_etiquette() -> pl.DataFrame:
    """Load and process etiquette-social-courtesies dataset."""
    dataset = load_dataset("hakeematyab/etiquette-social-courtesies")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": f"{safe_str(row.get('situation'))}"},
                {"role": "assistant", "content": safe_str(row.get("response"))}
            ],
            source="hakeematyab/etiquette-social-courtesies",
            metadata={
                "category": "etiquette",
                "etiquette_category": safe_str(row.get("category"))
            }
        )

    return process_dataset(df, process_row)

def load_coat_dataset() -> pl.DataFrame:
    """Load and process COAT dataset."""
    dataset = load_dataset("RussianNLP/coat")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": f"{safe_str(row.get('source'))}"},
                {"role": "assistant", "content": safe_str(row.get("target"))}
            ],
            source="RussianNLP/coat",
            metadata={
                "category": "text_correction",
                "correction_type": safe_str(row.get("type"))
            }
        )

    return process_dataset(df, process_row)

def load_russian_summaries() -> pl.DataFrame:
    """Load and process Mixed-Summarization-Dataset."""
    dataset = load_dataset("RussianNLP/Mixed-Summarization-Dataset")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": f"{safe_str(row.get('text'))}"},
                {"role": "assistant", "content": safe_str(row.get("summary"))}
            ],
            source="RussianNLP/Mixed-Summarization-Dataset",
            metadata={
                "category": "summarization",
                "text_source": safe_str(row.get("source"))
            }
        )

    return process_dataset(df, process_row)

def load_pikabu_stories() -> pl.DataFrame:
    """Load and process pikabu_text_norm dataset."""
    dataset = load_dataset("saarus72/pikabu_text_norm")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": f"{safe_str(row.get('title'))}"},
                {"role": "assistant", "content": safe_str(row.get("text"))}
            ],
            source="saarus72/pikabu_text_norm",
            metadata={
                "category": "creative_writing",
                "tags": row.get("tags", [])
            }
        )

    return process_dataset(df, process_row)

def load_coedit() -> pl.DataFrame:
    """Load and process Grammarly's coedit dataset."""
    dataset = load_dataset("grammarly/coedit")
    df = dataset["train"].to_pandas()

    def process_row(row):
        return create_conversation(
            messages=[
                {"role": "user", "content": f"{safe_str(row.get('source'))}"},
                {"role": "assistant", "content": safe_str(row.get("target"))}
            ],
            source="grammarly/coedit",
            metadata={
                "category": "text_improvement",
                "edit_type": safe_str(row.get("edit_type"))
            }
        )

    return process_dataset(df, process_row)

def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    datasets = {
        "chatgpt_prompts": load_awesome_chatgpt_prompts,
        "roleplay": load_roleplay_dataset,
        "herman_json": load_herman_json,
        "jailbreak": load_jailbreak_behaviors,
        "social_reasoning": load_social_reasoning,
        "hallucination": load_hallucination_dataset,
        "logic_puzzles": load_logic_puzzles,
        "brain_teasers": load_brain_teasers,
        "propositional_logic": load_propositional_logic,
        "numina_math": load_numina_math,
        "openmath_instruct": load_openmath_instruct,
        "etiquette": load_etiquette,
        "coat": load_coat_dataset,
        "russian_summaries": load_russian_summaries,
        "coedit": load_coedit,
        "pikabu_stories": load_pikabu_stories
    }

    all_data = []
    for name, loader_func in datasets.items():
        try:
            logger.info(f"Processing {name} dataset...")
            df = loader_func()

            # Validate schema
            required_cols = {"conversations", "source", "metadata"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            all_data.append(df)

            # Save individual dataset
            df.write_parquet(output_dir / f"{name}.parquet")
            logger.info(f"Saved {name} dataset with {len(df)} rows")

        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")

    # Combine all datasets
    if all_data:
        combined_df = pl.concat(all_data)
        combined_df.write_parquet(output_dir / "combined_datasets.parquet")
        logger.info(f"Saved combined dataset with {len(combined_df)} rows")

if __name__ == "__main__":
    main()
