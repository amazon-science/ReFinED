from dataclasses import dataclass
from typing import List


@dataclass
class AdditionalEntity:
    label: str
    aliases: List[str]
    description: str   # English
    entity_types: List[str]   # Wikidata qcodes
    entity_id: str  # Use A followed by a number for example "A100"
