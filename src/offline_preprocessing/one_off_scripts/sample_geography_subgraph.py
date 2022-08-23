from functools import lru_cache
from typing import Dict, List, FrozenSet, Set
import ujson as json
from tqdm.auto import tqdm


class ClassExplorer:
    def __init__(self, subclasses: Dict[str, Set[str]]):
        self.subclasses: Dict[str, Set[str]] = subclasses

    @lru_cache(maxsize=None)
    def explore_class_tree(self, qcode: str, explored_classes: FrozenSet[str]) -> FrozenSet[str]:
        """
        Recursively explores the class hierarchy (parent classes, parent of parents, etc.)
        Returns all of the explored classes (these are all impliable from the class provided as an argument (evi_class))
        :param qcode: class id for class to explore
        :param explored_classes: the classes impliable from class_id
        :return: a set of classes that are (indirect) direct ancestors of class_id
        """
        # This method will explore evi_class so add it to the explored_classes set to prevent repeating the work
        explored_classes = set(explored_classes)
        explored_classes.add(qcode)
        explored_classes.copy()

        # Base case: Evi class has no super classes so return the explored classes
        if qcode not in self.subclasses:
            return frozenset(explored_classes)

        # General case: Explore all unexplored super classes
        for super_class in self.subclasses[qcode]:
            if super_class not in explored_classes:
                explored_classes.add(super_class)
                explored_super_classes = self.explore_class_tree(
                    super_class, frozenset(explored_classes)
                )
                explored_classes.update(explored_super_classes)
        return frozenset(explored_classes)

    @lru_cache(maxsize=None)
    def get_implied_classes(
            self, direct_classes: FrozenSet[str], remove_self=False
    ) -> FrozenSet[str]:
        """
        From a set of (direct) classes this method will generate all of the classes that can be implied.
        When remove_self is True it means that a class cannot be implied from itself (but it can still be implied
        by other of the direct classes).
        :param direct_classes: the set of classes for implied classes to be generated from
        :param remove_self: when true a classes implication is not reflexive (e.g. human does not imply human)
        :return: set of classes that can be implied from direct_classes
        """
        if remove_self:
            all_implied_classes = set()
        else:
            all_implied_classes = set(direct_classes)

        # keep track of the classes that have been explored to prevent work from being repeated
        explored_classes = set()
        for direct_class in direct_classes:
            implied_classes = self.explore_class_tree(direct_class, frozenset(explored_classes))
            if remove_self:
                implied_classes = implied_classes - {direct_class}

            explored_classes.update(implied_classes)
            all_implied_classes.update(implied_classes)

        return frozenset(all_implied_classes)


geographically_in: Dict[str, Set[str]] = dict()
debug = False
with open('geography_subgraph_only.json', 'r') as f:
    for i, line in tqdm(enumerate(f), total=10000000, desc='Loading located_in dictionary'):
        line = json.loads(line)
        geographically_in[line['qcode']] = set(line['values'])
        if debug and i > 500000:
            break

class_explorer = ClassExplorer(subclasses=geographically_in)

for i, (qcode, direct_parents) in enumerate(geographically_in.items()):
    if i < 1000:
        print('qcode', qcode, 'direct_parents', direct_parents, 'implied parents',
              class_explorer.get_implied_classes(frozenset(direct_parents)))
    if debug and i > 100:
        break
