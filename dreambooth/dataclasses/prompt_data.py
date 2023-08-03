import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional


@dataclass
class PromptData:
    prompt: str = ""
    negative_prompt: str = ""
    instance_token: str = ""
    class_token: str = ""
    src_image: str = ""
    steps: int = 40
    scale: float = 7.5
    out_dir: str = ""
    seed: int = -1
    resolution: Tuple[int, int] = (512, 512)
    original_resolution: Tuple[int, int] = (512, 512)
    concept_index: int = 0
    is_class_image: bool = False
    weight: float = 1.0
    roi_rects: Optional[List[Tuple[int, int, int, int]]] = None
    mask_image: Optional[str] = None

    def __post_init__(self):
        if self.seed == -1:
            self.seed = int(random.randrange(0, 21474836147))

    @property
    def json(self):
        """
        get the json formated string
        """
        return json.dumps(asdict(self))
