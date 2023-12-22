from abc import ABC, abstractmethod
from random import sample
from typing import Tuple, List, Any, Mapping, Set, Dict, Optional
from collections import defaultdict


from refined.data_types.base_types import Span
from refined.utilities.general_utils import unique
from refined.resource_management.loaders import normalize_surface_form


class CandidateGenerator(ABC):

    @abstractmethod
    def get_candidates(self, *args, **kwargs) -> Tuple[List[Tuple[str, float]], Any]:
        """
        Generates a list of candidate entities (IDs) along with initial/prior scores.
        The
        :return: Returns a tuple. First item is a list of candidate entities (IDs) along with initial/prior scores.
                 The second item can be any object that can be used to maintain state across calls.
        """
        pass

    @abstractmethod
    def add_candidates_to_spans(self, spans: List[Span], **kwargs) -> Any:
        """
        Add candidate entities to spans in-place. Can return any object to maintain state.
        """
        pass


class CandidateGeneratorExactMatch(CandidateGenerator):

    def __init__(self, max_candidates: int, pem: Mapping[str, List[Tuple[str, float]]], human_qcodes: Set[str], combine: bool=True, language: str='None'):
        self.max_candidates = max_candidates
        self.pem = pem
        self.human_qcodes = human_qcodes
        self.mention2wikidataID: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.lang_title2wikidataID: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.language = language
        self.combine = combine
        
    def change_to_pem(self, qcode_link_counts):
        pem: Dict[str, Dict[str, float]] = defaultdict(dict)
        total_link_count = sum(qcode_link_counts.values())
        pem = dict(sorted([(qcode, link_count / total_link_count) for qcode, link_count in
                                         qcode_link_counts.items()], key=lambda x: x[1], reverse=True))
        return pem
    
    def mgenre_title_mapping(self,surface_form: str, candidate_entities,combine_two_mapping: bool=False):
        if (self.language,surface_form) in self.lang_title2wikidataID:
            q_id_select = self.lang_title2wikidataID[(self.language,surface_form)]
            candidate_entities = dict()
            for new_cand in list(q_id_select):
                candidate_entities[new_cand] = 1    
        
        if combine_two_mapping and surface_form in self.mention2wikidataID and self.language != 'ar':
            candidate_entities_m = self.change_to_pem(self.mention2wikidataID[surface_form])
            max_len = max(len(candidate_entities_m),self.max_candidates) 
            for new_cand in list(candidate_entities_m.items())[:max_len]:
                if new_cand[0] not in {cand for cand,pior in candidate_entities.items()}:
                    candidate_entities[new_cand[0]] = new_cand[1]
                else:
                    candidate_entities[new_cand[0]] = max(new_cand[1],candidate_entities[new_cand[0]])
        elif not combine_two_mapping and surface_form in self.mention2wikidataID:
            candidate_entities = self.change_to_pem(self.mention2wikidataID[surface_form])
   
        
        direct_cands = list(candidate_entities.items())
        direct_cands = sorted(direct_cands,key=lambda x: x[1], reverse=True)
        return direct_cands

    def get_candidates(
            self,
            surface_form: str,
            person_coref_ref: Dict[str, List[Tuple[str, float]]] = None,
            sample_k_candidates: Optional[int] = None,
            gold_qcode: Optional[str] = None
    ) -> Tuple[List[Tuple[str, float]], Dict[str, List[Tuple[str, float]]]]:
        """
        Generates a list of candidate entities for a surface form using a pem lookup. Person name coreference is done
        by copying pem values from longer person name to short mentions of person name e.g. "Donald Trump" and "Trump".
        :param surface_form: surface form to generate candidate for (does not need to be normalised)
        :param person_coref_ref: person co-reference dictionary (keeps track of person names and part of names)
        :param sample_k_candidates: randomly samples `k` candidates (hard, random, gold)
        :param gold_qcode: gold_qcode only required when `sample_k_candidates` is provided
        :return: (list of candidates with pem value, dictionary of person name to candidates with pem value).
        """
        # max_cands is only used to cap candidates when return
        max_cands = sample_k_candidates if sample_k_candidates is not None else self.max_candidates
        # the cap for pem value to be set when origin is person name co-reference
        person_coref_pem_cap = 0.80
        # minimum pem value to apply person name co-reference to
        person_coref_pem_min = 0.05
        if person_coref_ref is None:
            person_coref_ref: Dict[str, List[Tuple[str, float]]] = dict()

        surface_form_norm = normalize_surface_form(surface_form, remove_the=True)
        if surface_form_norm not in self.pem:
            candidate_entities = dict()
            direct_cands = self.mgenre_title_mapping(surface_form,candidate_entities=candidate_entities)
            if not direct_cands:
                return [("Q0", 0.0)] * max_cands, person_coref_ref
        else:
            # surface is in pem
            # direct candidates - means the surface form was directly in pem lookup
            cands_main = self.pem[surface_form_norm]  
            direct_cands = list(cands_main.items())
            
            if self.combine:
                direct_cands = self.mgenre_title_mapping(surface_form,candidate_entities=cands_main,combine_two_mapping=True)
 
        # add short names to person_coref for all people candidates
        person_short_names = surface_form_norm.split(" ")
        short_name_cands = []
        for qcode, pem_value in direct_cands:
            if qcode in self.human_qcodes and pem_value > person_coref_pem_min:
                short_name_cands.append((qcode, min(pem_value, person_coref_pem_cap)))
        if len(short_name_cands) > 0 and len(person_short_names) > 1:
            for short_name in person_short_names:
                person_coref_ref[short_name] = short_name_cands

        # check to see if surface form is a person name co-reference
        if surface_form_norm in person_coref_ref:
            indirect_cands = person_coref_ref[surface_form_norm]
            cands = list(
                unique(
                    lambda x: x[0],
                    sorted(direct_cands + indirect_cands, key=lambda x: x[1], reverse=True),
                )
            )
        else:
            cands = direct_cands
            
            
        if sample_k_candidates is not None:
            popular_negatives = sample_k_candidates // 2  # half are popular
            random_negatives = max(
                sample_k_candidates - popular_negatives - 1, 0
            )  # half are random (leave 1 space for gold candidate if there is no gold then pad candidate)

            assert gold_qcode is not None, "gold_qcode must be set when sample_k_candidate is set"
            # assuming it is already sorted TODO: check this

            negative_cands = [cand for cand in cands[:sample_k_candidates] if cand[0] != gold_qcode]
            # only add the gold_cand to the sample_k if it is in the top-k candidates
            gold_cand = []
            for cand in cands[:sample_k_candidates]: # (ReFinED's paper)
                if cand[0] == gold_qcode:
                    gold_cand = [cand]
                    break

            # popular negatives
            sampled_cands = negative_cands[:popular_negatives]

            # random negatives
            sampled_cands += sample(
                negative_cands[popular_negatives:],
                k=min(random_negatives, len(negative_cands[popular_negatives:])),
            )

            # gold candidate (will be added if it is present in the top-30 candidates) (ReFinED's paper)
            cands = gold_cand + sampled_cands

        cands = (cands + [("Q0", 0.0)] * max_cands)[:max_cands]
        return cands, person_coref_ref

    def add_candidates_to_spans(
            self,
            spans: List[Span],
            backward_coref: bool = False,
            person_coreference: Optional[Dict[str, List[Tuple[str, float]]]] = None,
            sample_k_candidates: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Add candidate entities to spans
        :param spans: list of spans
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherise will build person co-reference dictionary sequentially so only forward coref
                               occurs.
        :param person_coreference: person co-reference dictionary
        :param sample_k_candidates: randomly sample `k` candidates
        """
        if person_coreference is None:
            person_coreference: Dict[str, List[Tuple[str, float]]] = dict()
        if backward_coref:
            # pre-populate person co-reference dictionary so co-reference can occur backwards.
            for span in spans:
                _, person_coreference = self.get_candidates(
                    surface_form=span.text,
                    person_coref_ref=person_coreference,
                    sample_k_candidates=sample_k_candidates,
                    gold_qcode=span.gold_entity.wikidata_entity_id if sample_k_candidates else None
                )
        for span in spans:
            candidates_qcodes, person_coreference = self.get_candidates(
                surface_form=span.text,
                person_coref_ref=person_coreference,
                sample_k_candidates=sample_k_candidates,
                gold_qcode=span.gold_entity.wikidata_entity_id if sample_k_candidates else None
            )
            span.candidate_entities = candidates_qcodes
        return person_coreference
