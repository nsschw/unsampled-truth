from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import json

@dataclass(frozen=True)
class TrialResult:
    item_id: str
    persona_wording: Optional[str]
    task_instruction: str
    prompt: str
    option_symbol: str
    presented_order: Tuple[str, ...]
    anchor_mapping: Dict[str, str]
    probabilities: Dict[str, float]
    off_option_mass: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrialResult':
        data['presented_order'] = tuple(data['presented_order'])
        return cls(**data)
