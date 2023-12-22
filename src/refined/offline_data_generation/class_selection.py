import logging
import os
import sys
from collections import Counter
from typing import Dict, Any
from typing import Set

import requests
import ujson as json
from tqdm.auto import tqdm

from refined.doc_preprocessing.class_handler import ClassHandler
from refined.resource_management.loaders import normalize_surface_form, get_candidates, load_pem, load_subclasses, \
    load_instance_of, load_occuptations, load_sports, load_country, get_qcode_classes

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
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


def select_classes(resources_dir: str, add_class_every_n_pages: int = 5000, number_of_classes: int = 1400,
                   is_test: bool = False):
    pem = load_pem(os.path.join(resources_dir, 'wiki_pem.json'), is_test=is_test)
    subclasses, _ = load_subclasses(os.path.join(resources_dir, 'subclass_p279.json'), is_test=is_test)
    instance_of = load_instance_of(os.path.join(resources_dir, 'instance_of_p31.json'), is_test=is_test)
    occupations = load_occuptations(os.path.join(resources_dir, 'occupation_p106.json'), is_test=is_test)
    sports = load_sports(os.path.join(resources_dir, 'sport_p641.json'), is_test=is_test)
    country = load_country(os.path.join(resources_dir, 'country_p17.json'), is_test=is_test)
    class_explorer = ClassHandler(subclasses=subclasses,
                                  qcode_to_idx=None,
                                  qcode_idx_to_class_idx=None,  # TODO check it is fine to use None
                                  index_to_class=None)
    chosen_classes = set()
    non_zero_value = 5e-6 # to prevent divide by zero

    # add class
    chosen_classes.add('Q16889133')

    chosen_classes.update(download_common_wikidata_classes())
    
    with open(os.path.join(resources_dir, 'wikipedia_links_aligned.json'), 'r') as f: 
        good_class_all = []
        line_count = 0
        separated = 0
        num_ents = 0
        true_positive = 0
        false_positive = 0

        for line in tqdm(f, desc='Processing pages to select optimal classes for entity disambiguation'):
            line_count += 1
            if len(chosen_classes) > number_of_classes:
                os.rename(os.path.join(resources_dir, 'chosen_classes.txt.part'),
                          os.path.join(resources_dir, 'chosen_classes.txt'))
                return
            if (line_count + 1) % (add_class_every_n_pages * 1) == 0 and len(good_class_all) > 0:
                pop_precision = true_positive / (true_positive + false_positive + non_zero_value) * 100
                s_rate = separated / (num_ents + non_zero_value) * 100
                tqdm.write(f'Popularity precision {pop_precision}, No Popularity precision: {s_rate}')
                with open(os.path.join(resources_dir, 'chosen_classes.txt.part'), 'w') as f_out:
                    f_out.write('\n'.join([x for x in chosen_classes]))

            if (line_count + 1) % add_class_every_n_pages == 0 and len(good_class_all) > 0:
                chosen_qcode = Counter(good_class_all).most_common(1)[0][0]
                chosen_qcode_freq = Counter(good_class_all).most_common(1)[0][1]
                tqdm.write(f'Chosen class {chosen_qcode}, number of entities disambiguated with hit '
                           f'{chosen_qcode_freq}/{num_ents}, number of chosen classes {len(chosen_classes)}')
                chosen_classes.add(chosen_qcode)
                good_class_all = []
                separated = 0
                num_ents = 0
                true_positive = 0
                false_positive = 0

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
                    true_positive += 1
                else:
                    false_positive += 1
    
    os.rename(os.path.join(resources_dir, 'chosen_classes.txt.part'),
              os.path.join(resources_dir, 'chosen_classes.txt'))