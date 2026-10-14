from collections import defaultdict
from typing import List, Dict


def anchor_distribution(trials: List[dict]) -> Dict[str, float]:
    """
    Maps the option symbol probabilities of each trial onto their semantic anchors
    (e.g., 'A' -> 'Strongly Disagree') and averages the probability mass per anchor.
    Trials are the dicts read from the raw JSONL results.
    """
    if not trials:
        return {}

    anchor_sums = defaultdict(float)
    
    for trial in trials:
        for symbol, prob in trial["probabilities"].items():
            anchor_sums[trial["anchor_mapping"][symbol]] += prob
            
    num_trials = len(trials)
    # Average the probability mass per semantic anchor
    return {anchor: total / num_trials for anchor, total in sorted(anchor_sums.items())}
