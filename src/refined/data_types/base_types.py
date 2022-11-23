import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set


@dataclass
class Token:
    """
    This class represents a token in tokenized text.
    """

    text: str
    token_id: int
    start: int
    end: int


@dataclass
class Date:
    text: Optional[str] = None
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None

    known_format: bool = True

    def can_identify_format(self):

        if self.day is None or self.month is None:
            return True

        # If the date string contains any letters then assume we can identify format properly
        if bool(re.search(r"[a-zA-Z]", self.text)):
            return True

        # If the day is 12 or under, it is impossible to identify if this is a US format date (month first)
        if self.day <= 12:
            return False

        return True

    def get_doc_format(self) -> Optional[str]:
        if self.day is None or self.month is None:
            return None

        # If the date string contains any letters then does not give away format of number-only dates
        if bool(re.search(r"[a-zA-Z]", self.text)):
            return None

        # If day is 12 or under then this date doesn't reveal more general format
        if self.day <= 12:
            return None

        numbers_only = "".join([l for l in self.text if l.isnumeric()])

        if str(self.day) in numbers_only and numbers_only.index(str(self.day)) == 0:
            return "day_first"
        else:
            return "month_first"


@dataclass
class Entity:
    wikidata_entity_id: Optional[str] = None
    wikipedia_entity_title: Optional[str] = None
    human_readable_name: Optional[str] = None  # such as Wikipedia title or Wikidata label
    parsed_string: Optional[str] = None

    def __post_init__(self):
        if self.wikidata_entity_id == 'Q-1':
            self.wikidata_entity_id = None

    def __repr__(self):
        if len([v for v in vars(self).values() if v is not None]) > 0:
            return 'Entity(' + ', '.join([f'{var}={val}' for var, val in vars(self).items() if val is not None]) + ')'
        else:
            return 'Entity not linked to a knowledge base'


@dataclass
class Span:
    """
    Represents a span (entity-mention) of text in a document, `Doc`.
    """

    # basic information
    text: str  # text of the span (must match text defined by indices/offsets)
    start: int  # start character offset in `Doc`
    ln: int  # length of span (number of characters) in `Doc`

    # The document ID for the document where the span was found.
    doc_id: Optional[int] = None

    # gold entity (i.e. labelled entity in a dataset) used for model training
    gold_entity: Optional[Entity] = None

    # candidate entities (entities that the model considers)
    # can be manually provided (when providing spans to process_text())
    # or added by the library
    candidate_entities: Optional[List[Tuple[Entity, float]]] = None

    # results of model inference
    predicted_entity: Optional[Entity] = None
    entity_linking_model_confidence_score: Optional[float] = None
    top_k_predicted_entities: Optional[Tuple[List[Entity], float]] = None
    predicted_entity_types: Optional[List[Tuple[str, str, float]]] = None  # (type_id, type_label, confidence)
    coarse_type: Optional[str] = "MENTION"  # High level types such as (MENTION, DATE)
    coarse_mention_type: Optional[str] = None  # OntoNotes/spaCy types for mentions (ORG, LOC, PERSON)
    date: Optional[Date] = None  # if the span represents a date this object parses the date
    failed_class_check: Optional[bool] = None  # Indicates predicted class and actual entity class mismatch

    # can be used to filter candidates to a given set (optional)
    pruned_candidates: Optional[Set[str]] = None

    def __repr__(self) -> str:
        return str([self.text, self.predicted_entity, self.coarse_mention_type])
