import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from unsampled_truth.extractors.base import ProbabilityExtractor


class HuggingFaceExtractor(ProbabilityExtractor):
    def __init__(self, model: AutoModelForCausalLM, processor: AutoTokenizer, temperature: float = 1.0):
        self.model = model
        self.processor = processor
        self.temperature = temperature
        self.model.eval()

    def get_absolute_probabilities(self, prompt: str, options: List[str], persona_wording: str = None) -> Dict[str, float]:
        
        messages = [{"role": "user", "content": prompt}] if persona_wording is None else [
            {"role": "system", "content": persona_wording},
            {"role": "user", "content": prompt}
        ]
        base_prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking = False
        )

        base_ids = self.processor.encode(base_prompt_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        base_length = base_ids.shape[1]

        option_probs = []
        
        for option in options:
            full_text = base_prompt_text + option
            full_ids = self.processor.encode(full_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits
                
            shift_logits = logits[0, :-1, :] / self.temperature
            shift_labels = full_ids[0, 1:]
            
            option_logits = shift_logits[base_length - 1:]
            option_labels = shift_labels[base_length - 1:]
            
            probs_vocab = F.softmax(option_logits, dim=-1)
            token_probs = probs_vocab.gather(dim=-1, index=option_labels.unsqueeze(-1)).squeeze(-1)

            option_probs.append(token_probs.prod().item())

        off_option_mass = max(0.0, 1.0 - sum(option_probs))
        
        results = dict(zip(options, option_probs))
        results["Off-Option-Mass"] = off_option_mass

        return results


    def get_absolute_probabilities_single_token(self, prompt: str, options: List[str], persona_wording: str = None) -> Dict[str, float]:
        messages = [{"role": "user", "content": prompt}] if persona_wording is None else [
            {"role": "system", "content": persona_wording},
            {"role": "user", "content": prompt}
        ]
        base_prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking = False
        )

        # 1. Tokenize the base prompt
        input_ids = self.processor.encode(base_prompt_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        # 2. Execute a SINGLE forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)

        # 3. Isolate the logits for the absolute last token in the prompt sequence
        # Shape reduces from [batch, sequence_length, vocab_size] to [vocab_size]
        next_token_logits = outputs.logits[0, -1, :] / self.temperature

        # 4. Calculate the probability distribution across the entire vocabulary
        vocab_probs = F.softmax(next_token_logits, dim=-1)

        results = {}
        option_prob_sum = 0.0

        for option in options:
            # 5. Get the token ID for the option
            option_token_ids = self.processor.encode(option, add_special_tokens=False)

            if len(option_token_ids) != 1:
                raise ValueError(f"Optimization Failure: Option '{option}' tokenizes to {len(option_token_ids)} tokens. This function strictly requires 1-token options.")

            target_token_id = option_token_ids[0]

            # 6. Extract the specific probability mass for this token from the global distribution
            prob = vocab_probs[target_token_id].item()

            results[option] = prob
            option_prob_sum += prob

        # 7. Off-option mass is simply whatever mass wasn't assigned to explicit options
        results["Off-Option-Mass"] = max(0.0, 1.0 - option_prob_sum)
        return results