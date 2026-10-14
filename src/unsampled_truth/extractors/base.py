from typing import Protocol, List, Dict, Optional

class ProbabilityExtractor(Protocol):
    def get_absolute_probabilities(self, prompt: str, options: List[str], persona_wording: Optional[str] = None) -> Dict[str, float]:
        """
        Takes a formatted prompt and a list of valid token sequences.
        Returns a dictionary mapping options to their absolute sequence probability.
        """
        ...

    def get_absolute_probabilities_single_token(self, prompt: str, options: List[str], persona_wording: Optional[str] = None) -> Dict[str, float]:
        """
        Same as get_absolute_probabilities, restricted to options that are a single token.
        """
        ...
