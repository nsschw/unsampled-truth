import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor, as_completed

# Option symbol -> semantic anchor mapping
from unsampled_truth.metrics.anchors import anchor_distribution

# Define the ordinal mapping
ordinal_mapping = {
    "Strongly Disagree": 1.0,
    "Disagree": 2.0,
    "Neutral": 3.0,
    "Agree": 4.0,
    "Strongly Agree": 5.0
}

BASELINE_PERSONAS = ["antonia", "donovan", "kyle", "nicole", "renate"]

ORDINAL_KEYS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
ORDINAL_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]

# ---------------------------------------------------------
# DATA LOADING FUNCTIONS
# ---------------------------------------------------------

def load_matrix_trials(filepath: str, baseline_persona: str) -> list:
    """
    Loads trials and extracts the prompt component levels of each Prompt Variant Matrix cell.
    """
    trials = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Map option symbol probabilities onto semantic anchors
            scored_dist = anchor_distribution([data])
            
            # Skip empty distributions
            if sum(scored_dist.values()) == 0:
                continue

            # item_id has the form '<item>_<item wording>', e.g. 'BFI0_V1'
            item, item_wording = data["item_id"].split('_')
            task_instruction = data["task_instruction"]
            option_symbol = data["option_symbol"]
            persona_wording = data["persona_wording"]

            trials.append({
                "baseline_persona": baseline_persona,
                "persona_wording": persona_wording,
                "item": item,
                "item_wording": item_wording,
                "task_instruction": task_instruction,
                "option_symbol": option_symbol,
                "distribution": scored_dist
            })
            
    return trials

def load_all_model_data(base_dir: str, model_name: str) -> pd.DataFrame:
    """Helper to load and flatten the Prompt Variant Matrices once."""
    base_path = Path(base_dir)
    all_records = []
    
    for filepath in base_path.rglob("*_raw.jsonl"):
        parts = filepath.stem.split('_')
        if len(parts) < 2:
            continue
            
        baseline_persona = parts[0].capitalize()
        name = parts[1]

        if name.lower() != model_name.lower() or baseline_persona.lower() not in BASELINE_PERSONAS:
            continue
            
        all_records.extend(load_matrix_trials(filepath, baseline_persona))

    return pd.DataFrame(all_records)


def calculate_ss_within(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Vectorized calculation of strictly within-group sum of squares."""
    unique_labels = np.unique(labels)
    ss_within = 0.0
    
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        n_group = len(idx)
        if n_group > 1:
            # Extract sub-matrix for the group, square it, and sum the upper triangle
            sub_matrix = dist_matrix[np.ix_(idx, idx)]
            group_ss = np.sum(np.triu(sub_matrix**2, k=1))
            ss_within += group_ss / n_group
            
    return ss_within

def partition_variation(group_df: pd.DataFrame) -> dict:
    """Computes db-R2 using vectorized CDF-based Wasserstein distances."""
    N = len(group_df)
    if N < 2:
        return {}

    # 1. Extract Distributions
    probs = np.array([ [dist.get(k, 0.0) for k in ORDINAL_KEYS] for dist in group_df['distribution'] ])
    
    # Force each row to sum to 1.0 to guarantee valid Wasserstein distances
    row_sums = probs.sum(axis=1, keepdims=True)
    # Use np.divide with the 'where' argument to safely handle any absolute zero rows
    probs = np.divide(probs, row_sums, out=np.zeros_like(probs), where=row_sums!=0)

    # 2. Compute CDFs and Vectorized Distance Matrix
    cdfs = np.cumsum(probs, axis=1)
    
    # 1D Wasserstein on a uniform grid is exactly the L1 distance of CDFs.
    condensed_dist = pdist(cdfs, metric='cityblock')
    dist_matrix = squareform(condensed_dist)

    # 3. Extract Metadata as NumPy arrays for rapid indexing
    baseline_labels = group_df['baseline_persona'].to_numpy()
    wording_labels = group_df['persona_wording'].to_numpy() 
    
    crossed_axes = {
        'item_wording': group_df['item_wording'].to_numpy(),
        'task_instruction': group_df['task_instruction'].to_numpy(),
        'option_symbol': group_df['option_symbol'].to_numpy()
    }

    # 4. Total variation (SS_T)
    ss_total = np.sum(condensed_dist**2) / N
    if ss_total == 0:
        return {axis: 0.0 for axis in ['baseline_persona', 'persona_wording'] + list(crossed_axes.keys())}

    db_r2 = {}

    # 5a. Nested: each persona wording belongs to exactly one baseline persona, so persona wording
    #   is credited only with the variation it explains beyond the baseline persona.
    ss_within_baseline = calculate_ss_within(dist_matrix, baseline_labels)
    ss_within_wording = calculate_ss_within(dist_matrix, wording_labels)

    db_r2['baseline_persona'] = max(0.0, 1.0 - (ss_within_baseline / ss_total))
    db_r2['persona_wording'] = max(0.0, (ss_within_baseline - ss_within_wording) / ss_total)

    # 5b. Crossed components: db-R2 = 1 - SS_W(k) / SS_T
    for axis, labels in crossed_axes.items():
        ss_within_crossed = calculate_ss_within(dist_matrix, labels)
        db_r2[axis] = max(0.0, 1.0 - (ss_within_crossed / ss_total))

    return db_r2


def _process_single_group(args: tuple) -> dict:
    """Helper function for ProcessPoolExecutor mapping."""
    item_id, group_df = args
    db_r2_scores = partition_variation(group_df)
    if db_r2_scores:
        db_r2_scores['item'] = item_id
    return db_r2_scores

def partition_all_items(df: pd.DataFrame, max_workers: int = None) -> pd.DataFrame:
    """Executes distance-based variation partitioning utilizing multiprocessing."""
    results = []
    
    # Prepare tuples of (item, dataframe_slice)
    groups = [(name, group) for name, group in df.groupby('item')]
    
    # Default to available CPU cores minus 1 to leave overhead for the OS
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_group, g): g for g in groups}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(groups), desc="Processing Items", leave=False):
            res = future.result()
            if res:
                results.append(res)
                
    return pd.DataFrame(results)


if __name__ == "__main__":
    data_dir = "../../outputs/raw_results/"

    model_list = [
        "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", 
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", 
        "gemma-4-E2B-it", "gemma-4-E4B-it",
        "Ministral-3-3B-Instruct-2512-BF16", "Ministral-3-8B-Instruct-2512-BF16", "Ministral-3-14B-Instruct-2512-BF16"
    ]

    all_results = pd.DataFrame()
    
    # Ensure output directory exists before saving
    Path("../../outputs/tables").mkdir(parents=True, exist_ok=True)

    for model in tqdm.tqdm(model_list, desc="Processing Models"):
        df = load_all_model_data(data_dir, model)
        
        # Guard clause if model data isn't found
        if df.empty:
            continue
        
        # Map each persona variant to a common label 
        for persona in df['baseline_persona'].unique():
            variant_mapping = {v: f"V{idx}" for idx, v in enumerate(df[df['baseline_persona'] == persona]['persona_wording'].unique())}
            df.loc[df['baseline_persona'] == persona, 'persona_wording'] = df[df['baseline_persona'] == persona]['persona_wording'].map(variant_mapping)

        df["persona_wording"] = df["persona_wording"] + "_" + df["baseline_persona"]
        
        # Execute the optimized multiprocessing flow
        df_partition = partition_all_items(df)
        df_partition['model'] = model
        
        all_results = pd.concat([all_results, df_partition], ignore_index=True)
        all_results.to_csv("../../outputs/tables/variation_partitioning.csv", index=False)