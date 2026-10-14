import argparse
import yaml
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Mistral3ForConditionalGeneration
from unsampled_truth.extractors.hf_extractor import HuggingFaceExtractor
from unsampled_truth.pipelines.prompt_variant_matrix import run_prompt_variant_matrix
from unsampled_truth.utils.gpu import get_optimal_gpu
import torch

def main():
    parser = argparse.ArgumentParser(description="Run LLM Inference Sweep")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Boot Hardware
    processor = AutoTokenizer.from_pretrained(cfg['model_id']) if "mistralai" not in cfg['model_id'] else AutoTokenizer.from_pretrained(cfg['model_id'], fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], device_map=f"cuda:{get_optimal_gpu()}", dtype="auto") if "mistralai" not in cfg['model_id'] else Mistral3ForConditionalGeneration.from_pretrained(cfg['model_id'], device_map=f"cuda:{get_optimal_gpu()}", dtype="auto")
    extractor = HuggingFaceExtractor(model=model, processor=processor)

    # Load the Option Symbol Registry
    registry_path = Path("data/option_symbols.json")
    with open(registry_path, "r", encoding="utf-8") as f:
        option_symbol_registry = json.load(f)

    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg['experiment_name']}_raw.jsonl"
    
    # Overwrite previous results of this experiment so re-runs never duplicate rows;
    # results are still streamed to disk continuously
    if out_file.exists():
        print(f"Overwriting existing results: {out_file}")
    with open(out_file, "w") as f:
        
        # 2. Loop through every questionnaire file
        for questionnaire_path in cfg['questionnaires']:
            print(f"Loading Questionnaire: {questionnaire_path}")
            with open(questionnaire_path, "r", encoding="utf-8") as d_file:
                items_list = json.load(d_file)["items"]
                
            # 3. Loop through every option symbol KEY requested in the config
            for option_symbol in cfg['option_symbols']:
                print(f"  -> Applying option symbol: {option_symbol}")
                
                # Look up the option symbol set from the registry
                if option_symbol not in option_symbol_registry:
                    raise ValueError(f"Option symbol '{option_symbol}' not found in data/option_symbols.json")
                    
                option_symbol_set = option_symbol_registry[option_symbol]
                
                # Run the Prompt Variant Matrix for this option symbol set
                trials = run_prompt_variant_matrix(
                    items=items_list,
                    option_symbol=option_symbol,
                    option_symbol_set=option_symbol_set,
                    single_token = option_symbol_set.get('single_token', False),
                    extractor=extractor,
                    task_instructions=cfg['task_instructions'],
                    persona_wordings=cfg.get('persona_wordings', [None]),
                    full_permutation=cfg.get('full_permutation', True)
                )
                
                # Stream results to disk
                for trial in trials:
                    f.write(trial.to_json() + "\n")

if __name__ == "__main__":
    main()