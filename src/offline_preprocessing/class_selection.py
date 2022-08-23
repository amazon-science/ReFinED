import argparse
import logging
import os
import sys
from collections import Counter
from functools import lru_cache
from typing import Dict, Any, FrozenSet
from typing import Set

import requests
import ujson as json
from tqdm.auto import tqdm

from utilities.lookup_utils import normalize_surface_form, get_candidates, load_pem, load_subclasses, \
    load_instance_of, load_occuptations, load_sports, load_country, get_qcode_classes

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOG = logging.getLogger(__name__)


url = 'https://query.wikidata.org/sparql'
query_relation_object_classes = """
SELECT ?class (COUNT(?subj) as ?cnt)
WHERE
{
     ?subj p:P2302 ?statement.
     ?statement ps:P2302 wd:Q21503250.  # replace with Q21510865 for value type constraints Q21503250 value
     ?statement pq:P2308 ?class.
     ?statement pq:P2309 wd:Q21503252.
     FILTER NOT EXISTS { ?subj wdt:P31 wd:Q19847637}
     FILTER NOT EXISTS {?class wdt:P279 wd:Q17442446}
     FILTER NOT EXISTS {?class wdt:P279 wd:Q15138389}
     SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

}
GROUP BY ?class
ORDER BY DESC(?cnt)
LIMIT 100
"""

query_relation_subject_classes = """
SELECT ?class (COUNT(?subj) as ?cnt)
WHERE
{
     ?subj p:P2302 ?statement.
     ?statement ps:P2302 wd:Q21510865.  # replace with Q21510865 for value type constraints Q21503250 value
     ?statement pq:P2308 ?class.
     ?statement pq:P2309 wd:Q21503252.
     FILTER NOT EXISTS { ?subj wdt:P31 wd:Q19847637}
     FILTER NOT EXISTS {?class wdt:P279 wd:Q17442446}
     FILTER NOT EXISTS {?class wdt:P279 wd:Q15138389}
     SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

}
GROUP BY ?class
ORDER BY DESC(?cnt)
LIMIT 100
"""


def download_common_wikidata_classes() -> Set[str]:
    object_classes_res = requests.get(url, params={'format': 'json', 'query': query_relation_object_classes}).json()
    subject_classes_res = requests.get(url, params={'format': 'json', 'query': query_relation_subject_classes}).json()
    obj_classes = {x['class']['value'][31:] for x in object_classes_res['results']['bindings']}
    subj_classes = {x['class']['value'][31:] for x in subject_classes_res['results']['bindings']}
    return obj_classes | subj_classes


def ent_good_classes(entity: Dict[str, Any], pem, occupations, sports, country, instance_of, class_explorer, subclasses,
                     already_chosen: Set[str] = frozenset()):
    surface_form = normalize_surface_form(entity['surface_form'])

    # TODO: consider randomly selected candidate instead see if it makes a difference
    gold_qcode = entity['qcode']
    qcodes = [qcode for qcode in get_candidates(pem, surface_form).keys()][:11]
    good_classes: Set[str] = get_qcode_classes(gold_qcode, occupations=occupations, sports=sports, country=country,
                                               instance_of=instance_of, class_explorer=class_explorer,
                                               subclasses=subclasses)

    good_classes_already_chosen = good_classes & already_chosen
    good_classes -= already_chosen

    candidate_qcodes = [qcode for qcode in qcodes if not qcode == gold_qcode][:10]

    a_candidate_has_class = False
    already_separated = True
    conflicting_qcodes = set()
    conflicting_qcodes.add(gold_qcode)

    for candidate_qcode in candidate_qcodes:
        candidate_classes = get_qcode_classes(candidate_qcode, occupations=occupations, sports=sports, country=country,
                                              instance_of=instance_of, class_explorer=class_explorer,
                                              subclasses=subclasses)
        if len(candidate_classes) > 0:
            a_candidate_has_class = True
        #         if len(good_classes_already_chosen - candidate_classes) > 0:
        if not (already_chosen & candidate_classes) == (already_chosen & get_qcode_classes(gold_qcode,
                                                                                           occupations=occupations,
                                                                                           sports=sports,
                                                                                           country=country,
                                                                                           instance_of=instance_of,
                                                                                           class_explorer=class_explorer,
                                                                                           subclasses=subclasses)):
            # already separated
            continue

        # candidate shares all classes (from chosen classes) with gold qcode
        conflicting_qcodes.add(candidate_qcode)
        already_separated = False
        candidate_classes -= good_classes_already_chosen
        if len(candidate_classes) > 0:
            good_classes -= candidate_classes

    if len(candidate_qcodes) > 1 and a_candidate_has_class and not already_separated:
        return good_classes, already_separated, conflicting_qcodes

    # already_separated and a_candidate_has_class
    return set(), already_separated, conflicting_qcodes


class ClassExplorer:
    def __init__(self, subclasses):
        self.subclasses = subclasses

    @lru_cache(maxsize=None)
    def explore_class_tree(self, qcode: str, explored_classes: FrozenSet[str]) \
            -> FrozenSet[str]:
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
                explored_super_classes = self.explore_class_tree(super_class, frozenset(explored_classes))
                explored_classes.update(explored_super_classes)
        return frozenset(explored_classes)

    @lru_cache(maxsize=None)
    def get_implied_classes(self, direct_classes: FrozenSet[str], remove_self=False) -> FrozenSet[str]:
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


def select_classes(resources_dir: str, add_class_every_n_pages: int = 5000, number_of_classes: int = 1500,
                   is_test: bool = False):
    pem = load_pem(os.path.join(resources_dir, 'wiki_pem.json'), is_test=is_test)
    subclasses, _ = load_subclasses(os.path.join(resources_dir, 'subclass_p279.json'), is_test=is_test)
    instance_of = load_instance_of(os.path.join(resources_dir, 'instance_of_p31.json'), is_test=is_test)
    occupations = load_occuptations(os.path.join(resources_dir, 'occupation_p106.json'), is_test=is_test)
    sports = load_sports(os.path.join(resources_dir, 'sport_p641.json'), is_test=is_test)
    country = load_country(os.path.join(resources_dir, 'country_p17.json'), is_test=is_test)
    class_explorer = ClassExplorer(subclasses)

    with open(os.path.join(resources_dir, 'wikipedia_links_aligned.json'), 'r') as f:
        good_class_all = []
        i = 0
        chosen_classes = set()

        # add class
        chosen_classes.add('Q16889133')

        chosen_classes.update(download_common_wikidata_classes())

        separated = 0
        num_ents = 0

        tp_p = 0
        fp_p = 0
        for line in tqdm(f, desc='Processing pages to select optimal classes for entity disambiguation'):
            i += 1
            if (i + 1) % (add_class_every_n_pages * 1) == 0:
                pop_precision = tp_p / (tp_p + fp_p + 5e-6) * 100
                s_rate = separated / (num_ents + 1e-6) * 100
                tqdm.write(f'Popularity precision {pop_precision}, No Popularity precision: {s_rate}')
                with open(os.path.join(resources_dir, 'chosen_classes.txt.part'), 'w') as f_out:
                    f_out.write('\n'.join([x for x in chosen_classes]))

            if (i + 1) % add_class_every_n_pages == 0:
                chosen_qcode = Counter(good_class_all).most_common(1)[0][0]
                chosen_qcode_freq = Counter(good_class_all).most_common(1)[0][1]
                tqdm.write(f'Chosen class {chosen_qcode}, number of entities disambiguated with hit '
                           f'{chosen_qcode_freq}/{num_ents}, number of chosen classes {len(chosen_classes)}')
                chosen_classes.add(chosen_qcode)
                good_class_all = []
                separated = 0
                num_ents = 0
                tp_p = 0
                fp_p = 0

            if i > (add_class_every_n_pages * number_of_classes):
                break
            line = json.loads(line)
            for ent in line['hyperlinks_clean']:
                good_classes, already_sep, conflicting_qcodes = ent_good_classes(ent, already_chosen=chosen_classes,
                                                                                 pem=pem,
                                                                                 occupations=occupations,
                                                                                 instance_of=instance_of,
                                                                                 country=country,
                                                                                 sports=sports,
                                                                                 class_explorer=class_explorer,
                                                                                 subclasses=subclasses)
                num_ents += 1
                if already_sep:
                    separated += 1
                good_class_all.extend(good_classes)

                gold_qcode = ent['qcode']
                cands = list(get_candidates(pem, ent['surface_form']).items())[:10]
                cands = [x[0] for x in cands if x[0] in conflicting_qcodes]
                if len(cands) > 0 and cands[0] == gold_qcode:
                    tp_p += 1
                else:
                    fp_p += 1
        os.rename(os.path.join(resources_dir, 'chosen_classes.txt.part'),
                  os.path.join(resources_dir, 'chosen_classes.txt'))


def main():
    parser = argparse.ArgumentParser(description='Process cleaned Wikipedia, extract links, merge files.')
    parser.add_argument(
        "--pem_file",
        type=str,
        default='output/wiki_pem.json',
        help="File path for pem lookup created from Wikipedia links."
    )
    parser.add_argument(
        "--instance_of_file",
        type=str,
        default='output/instance_of_p31.json',
        help="File path for Wikidata instance_of (classes) file."
    )
    parser.add_argument(
        "--subclasses_file",
        type=str,
        default='output/subclass_p279.json',
        help="File path for Wikidata subclasses file."
    )
    parser.add_argument(
        "--occupation_file",
        type=str,
        default='output/occupation_p106.json',
        help="File path for Wikidata occupation file."
    )
    parser.add_argument(
        "--sport_file",
        type=str,
        default='output/sport_p641.json',
        help="File path for Wikidata sport file."
    )
    parser.add_argument(
        "--country_file",
        type=str,
        default='output/country_p17.json',
        help="File path for Wikidata country file."
    )

    parser.add_argument(
        "--wikitext_aligned_file",
        type=str,
        default='output_aligned/wikitext_aligned_wd_shuf.json',
        help="File path for cleaned wikipedia text with links extracted and aligned to Wikidata."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output_class_selection',
        help="Directory where the output will be stored"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--add_class_every_n_pages",
        type=int,
        default=5000,
        help="Add class after processing every n pages"
    )
    parser.add_argument(
        "--number_pages_to_process",
        type=int,
        default=6000000,
        help="End program after processing this many pages (max is 6M)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="mode for testing (only processes first 500 lines)"
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip('/')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pem = load_pem()
    subclasses, _ = load_subclasses()
    instance_of = load_instance_of()
    occupations = load_occuptations()
    sports = load_sports()
    country = load_country()
    class_explorer = ClassExplorer(subclasses)

    with open(args.wikitext_aligned_file, 'r') as f:
        good_class_all = []
        i = 0
        chosen_classes = set()
        separated = 0
        num_ents = 0

        tp_p = 0
        fp_p = 0
        for line in tqdm(f, desc='Processing pages to select optimal classes for entity disambiguation'):
            i += 1
            if (i + 1) % (args.add_class_every_n_pages * 5) == 0:
                pop_precision = tp_p / (tp_p + fp_p + 5e-6) * 100
                s_rate = separated / (num_ents + 1e-6) * 100
                LOG.info(f'Popularity precision {pop_precision}, No Popularity precision: {s_rate}')
                with open(f'{args.output_dir}/chosen_classes.txt', 'w') as output:
                    output.write('\n'.join([x for x in chosen_classes]))

            if (i + 1) % args.add_class_every_n_pages == 0:
                chosen_qcode = Counter(good_class_all).most_common(1)[0][0]
                chosen_classes.add(chosen_qcode)
                good_class_all = []
                separated = 0
                num_ents = 0
                tp_p = 0
                fp_p = 0
            if i > args.number_pages_to_process:
                break
            line = json.loads(line)
            for ent in line['hyperlinks_clean']:
                good_classes, already_sep, conflicting_qcodes = ent_good_classes(ent, already_chosen=chosen_classes,
                                                                                 pem=pem,
                                                                                 occupations=occupations,
                                                                                 instance_of=instance_of,
                                                                                 country=country,
                                                                                 sports=sports,
                                                                                 class_explorer=class_explorer,
                                                                                 subclasses=subclasses)
                num_ents += 1
                if already_sep:
                    separated += 1
                good_class_all.extend(good_classes)

                gold_qcode = ent['qcode']
                cands = list(get_candidates(pem, ent['surface_form']).items())[:10]
                cands = [x[0] for x in cands if x[0] in conflicting_qcodes]
                if len(cands) > 0 and cands[0] == gold_qcode:
                    tp_p += 1
                else:
                    fp_p += 1


if __name__ == '__main__':
    main()
