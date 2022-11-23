from typing import FrozenSet, List, Tuple

import torch

from refined.data_types.base_types import Span


class ClassHandler:
    def __init__(self, subclasses, index_to_class, qcode_idx_to_class_idx, qcode_to_idx):
        # class -> [parent_classes]
        # Dict[str, List[str]]
        self.subclasses = subclasses
        self.index_to_class = index_to_class
        self.qcode_idx_to_class_idx = qcode_idx_to_class_idx
        self.qcode_to_idx = qcode_to_idx

        # workaround - @lru_cache does not work with instances methods so basic implementation is done here
        self._MAX_CACHE_ITEMS = 10000000
        self._prune_classes_cache = {}
        self._get_implied_classes_cache = {}

    def _get_implied_classes(
            self, direct_classes: FrozenSet[str], remove_self=True
    ) -> FrozenSet[str]:
        """
        From a set of (direct) classes this method will generate all of the classes that can be implied.
        When remove_self is True it means that a class cannot be implied from itself (but it can still be implied
        by other of the direct classes).
        :param direct_classes: the set of classes for implied classes to be generated from
        :param remove_self: when true a classes implication is not reflexive (e.g. human does not imply human)
        :return: set of classes that can be implied from direct_classes
        """
        cache_key = (direct_classes, remove_self)
        if cache_key in self._get_implied_classes_cache:
            return self._get_implied_classes_cache[cache_key]

        if remove_self:
            all_implied_classes = set()
        else:
            all_implied_classes = set(direct_classes)

        # keep track of the classes that have been explored to prevent work from being repeated
        explored_classes = set()
        for direct_class in direct_classes:
            implied_classes = self._explore_class_tree(direct_class, frozenset(explored_classes))
            if remove_self:
                implied_classes = implied_classes - {direct_class}

            explored_classes.update(implied_classes)
            all_implied_classes.update(implied_classes)

        result = frozenset(all_implied_classes)
        self._get_implied_classes_cache[cache_key] = result
        if len(self._get_implied_classes_cache) > self._MAX_CACHE_ITEMS:
            self._get_implied_classes_cache.popitem()
        return result

    def _explore_class_tree(
            self, class_id: str, explored_classes: FrozenSet[str]
    ) -> FrozenSet[str]:
        """
        Recursively explores the class hierarchy (parent classes, parent of parents, etc.)
        Returns all the explored classes (these are all impliable from the class provided as an argument (class_id))
        :param class_id: class id for class to explore
        :param explored_classes: the classes impliable from class_id
        :return: a set of classes that are (indirect) direct ancestors of class_id
        """
        # This method will explore evi_class so add it to the explored_classes set to prevent repeating the work
        explored_classes = set(explored_classes)
        explored_classes.add(class_id)
        explored_classes.copy()

        # Base case: the class has no super classes so return the explored classes
        if class_id not in self.subclasses:
            return frozenset(explored_classes)

        # General case: Explore all unexplored super classes
        for super_class in self.subclasses[class_id]:
            if super_class not in explored_classes:
                explored_classes.add(super_class)
                explored_super_classes = self._explore_class_tree(
                    super_class, frozenset(explored_classes)
                )
                explored_classes.update(explored_super_classes)
        return frozenset(explored_classes)

    def prune_classes(self, class_ids: FrozenSet[str]) -> FrozenSet[str]:
        """
        Prune classes that are inferred from other provided classes.
        Note that this also filters country and sport relation as well.
        :param class_ids: a set of classes
        :return: set of fine-grained classes that are not inferred by each other (i.e. classes are different subtrees)
        """
        if class_ids in self._prune_classes_cache:
            return self._prune_classes_cache[class_ids]
        classes = frozenset({class_id for class_id in class_ids if "<" not in class_id})
        implied_classes = self._get_implied_classes(classes, remove_self=True)
        result = frozenset({str(x) for x in classes - implied_classes})
        self._prune_classes_cache[class_ids] = result
        if len(self._prune_classes_cache) > self._MAX_CACHE_ITEMS:
            self._prune_classes_cache.popitem()
        return result

    def class_check_span(self, span_to_check: Span):
        if span_to_check.predicted_entity is not None \
                and span_to_check.predicted_entity.wikidata_entity_id is not None:
            predicted_entity = span_to_check.predicted_entity.wikidata_entity_id
            predicted_classes = {qcode for qcode, label, conf in span_to_check.predicted_entity_types}
            entity_classes = self.get_classes_idx_for_qcode_batch([predicted_entity])
            class_indices = entity_classes[entity_classes != 0].numpy().tolist()
            entity_classes = [self.index_to_class[idx] for idx in class_indices]
            entity_classes = [c for c in entity_classes if "<" not in c]
            entity_classes = self._get_implied_classes(frozenset(entity_classes), remove_self=False)
            if len(set(predicted_classes) & entity_classes) > 0 or len(entity_classes) == 0:
                span_to_check.failed_class_check = False
            else:
                span_to_check.entity_linking_model_confidence_score = -1.0
                span_to_check.failed_class_check = True

    def class_check_spans(self, spans_to_check: List[Span]):
        for span_to_check in spans_to_check:
            self.class_check_span(span_to_check)

    def get_classes_idx_for_qcode_batch(
            self, qcodes: List[str], shape: Tuple[int, ...] = None
    ) -> torch.Tensor:
        """
        Retrieves all of the classes indices for the qcodes (from various relations used to construct the lookup).
        :param qcodes: qcodes
        :param shape: shape/size of tensor returned
        :return: long tensor with shape (num_ents, self.max_num_classes) (qcode index 0 is used for padding all classes)
        """
        result = torch.tensor(
            self.qcode_idx_to_class_idx[
                [self.qcode_to_idx[qcode] if qcode in self.qcode_to_idx else 0 for qcode in qcodes]
            ],
            dtype=torch.long,
        )
        if shape is not None:
            return result.view(shape)
        else:
            return result
