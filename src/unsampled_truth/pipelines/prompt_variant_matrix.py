import itertools
from typing import List, Iterator, Optional
import math
from tqdm import tqdm
from unsampled_truth.core.types import TrialResult
from unsampled_truth.extractors.base import ProbabilityExtractor

def run_prompt_variant_matrix(
    items: List[dict],
    option_symbol: str,
    option_symbol_set: dict,
    single_token: bool,
    extractor: ProbabilityExtractor,
    task_instructions: List[str],
    persona_wordings: List[Optional[str]],
    full_permutation: bool = True          # True: all anchor orderings, False: forward order only
) -> Iterator[TrialResult]:
    
    if not persona_wordings:
        persona_wordings = [None]
        
    symbols = option_symbol_set['symbols'][:len(items[0]["semantic_anchors"])]
    
    # Calculate anchor orderings for the progress bar total
    if full_permutation:
        num_anchor_perms = math.factorial(len(items[0]["semantic_anchors"]))
    else:
        num_anchor_perms = 1               # Forward order only
        
    total_perms_per_item = num_anchor_perms * len(task_instructions) * len(persona_wordings)
    
    for item_data in tqdm(items, desc="Questionnaire Items", position=0):
        item_id = item_data["item_id"]
        item_text = item_data["text"]
        
        # Determine the anchor orderings to iterate over
        base_anchors = item_data["semantic_anchors"]
        if full_permutation:
            anchor_orderings = list(itertools.permutations(base_anchors))
        else:
            anchor_orderings = [tuple(base_anchors)]
        
        with tqdm(total=total_perms_per_item, desc=f"Prompt Variant Matrix {item_id}", position=1, leave=False) as inner_pbar:
            
            for persona_wording in persona_wordings:
                for task_instruction in task_instructions:
                    
                    # Iterate only over the defined anchor orderings
                    for anchor_perm in anchor_orderings:
                        
                        # Zip the static tokens with the permuted anchors
                        anchor_mapping = dict(zip(symbols, anchor_perm))

                        prompt = task_instruction
                        if r"{range}" in prompt:
                            prompt = prompt.replace("{range}", option_symbol_set['range'])
                        if r"{option_name}" in prompt:
                            prompt = prompt.replace("{option_name}", option_symbol_set['option_name'])

                        prompt = prompt.format(
                            item=item_text, 
                            option_mapping=", ".join([f"{tkn} = {anchor_mapping[tkn]}" for tkn in symbols])
                        )
                        
                        if single_token:
                            probs = extractor.get_absolute_probabilities_single_token(prompt=prompt, options=symbols, persona_wording=persona_wording)
                        else:
                            probs = extractor.get_absolute_probabilities(prompt=prompt, options=symbols, persona_wording=persona_wording)

                        yield TrialResult(
                            item_id=item_id,
                            persona_wording=persona_wording,
                            task_instruction=task_instruction,
                            prompt=prompt,
                            option_symbol=option_symbol,
                            presented_order=tuple(symbols),         # Symbols are always presented in their static order
                            anchor_mapping=anchor_mapping,
                            probabilities={k: v for k, v in probs.items() if k != "Off-Option-Mass"},
                            off_option_mass=probs.get("Off-Option-Mass", 0.0)
                        )
                        
                        inner_pbar.update(1)