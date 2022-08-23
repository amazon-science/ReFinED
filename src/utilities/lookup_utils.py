import bz2
import io
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple
from urllib.parse import unquote

import smart_open
from tqdm.auto import tqdm
from unidecode import unidecode
from utilities.general_utils import get_logger

import ujson

LOG = get_logger(__name__)

# TODO: add function to download resources if not present same with training script so no setup s3 sync is required


def load_qcode_to_idx(filename: str, is_test: bool = False) -> Dict[str, int]:
    LOG.info("Loading qcode_to_idx")
    qcode_to_idx: Dict[str, int] = dict()
    line_num = 0
    with smart_open.smart_open(filename, "r") as f:
        if "s3" in filename:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=6000000, desc="Loading qcode_to_idx"):
            line = ujson.loads(line)
            qcode = line["qcode"]
            idx = line["index"]
            qcode_to_idx[qcode] = idx
            line_num += 1
            if is_test and line_num > 1000:
                break

    return qcode_to_idx


def load_descriptions(
    filepath: str, qcodes: Optional[Set[str]] = None, is_test: bool = False,
    keep_all_entities: bool = True
) -> Dict[str, str]:
    LOG.info("Loading descriptions")
    descriptions = dict()
    with smart_open.smart_open(filepath, "r") as f:
        if "s3" in filepath:
            f = io.StringIO(f.read())
        i = 0
        for line in tqdm(f, total=80000000, desc="Loading descriptions"):
            line = ujson.loads(line)
            qcode = line["qcode"]
            if qcodes is None or qcode in qcodes or keep_all_entities:
                descriptions[qcode] = line["values"]
            i += 1
            if is_test and i > 1000:
                return descriptions
    return descriptions


def load_instance_of(file_path: str, is_test: bool = False) -> Dict[str, Set[str]]:
    LOG.info("Loading instance_of (classes)")
    instance_of = dict()
    line_num = 0
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=80e6, desc="Loading Wikidata instance_of (classes)"):
            line = ujson.loads(line)
            instance_of[line["qcode"]] = set(line["values"])
            line_num += 1
            if is_test and line_num > 10000:
                break
    LOG.info(f"Loaded instance_of, size = {len(instance_of)}")
    return instance_of


def load_subclasses(file_path: str, is_test: bool = False):
    LOG.info("Loading subclasses")
    subclasses = dict()
    subclasses_reversed = defaultdict(set)
    line_num = 0
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=2e6, desc="Loading Wikidata subclasses"):
            line = ujson.loads(line)
            subclasses[line["qcode"]] = set(line["values"])
            for superclass in line["values"]:
                subclasses_reversed[superclass].add(line["qcode"])
            line_num += 1
            if is_test and line_num > 10000:
                break
    LOG.info(f"Loaded subclasses, size = {len(subclasses)}")
    return subclasses, subclasses_reversed


def load_redirects(file_path: str, is_test: bool = False):
    backslash = "\\"
    double_backslash = backslash * 2
    unescape_quotes = (
        lambda string: string.replace(double_backslash, "")
        .replace(backslash, "")
        .replace("&amp;", "&")
    )
    LOG.info("Loading redirects")
    redirects = dict()
    line_num = 0
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=9e6, desc="Loading Wikipedia page redirects"):
            line = ujson.loads(line)
            redirects[unescape_quotes(line["wiki_title"])] = unescape_quotes(line["dest_title"])
            line_num += 1
            if is_test and line_num > 10000:
                break
    LOG.info(f"Loaded redirects, size = {len(redirects)}")
    return redirects


def load_wikipedia_to_qcode(file_path: str, is_test: bool = False):
    LOG.info("Loading enwiki sitelinks")
    enwiki_to_qcode = dict()
    line_num = 0
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=8e6, desc="Loading enwiki sitelinks"):
            line = ujson.loads(line)
            enwiki_to_qcode[line["values"].replace(" ", "_")] = line["qcode"]
            line_num += 1
            if is_test and line_num > 10000:
                break
    LOG.info(f"Loaded enwiki_to_qcode, size = {len(enwiki_to_qcode)}")
    return enwiki_to_qcode


def load_disambiguation_qcodes(file_path: str, is_test: bool = False):
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        disambiguation_qcodes: Set[str] = {l.rstrip("\n") for l in f.readlines()}
    return disambiguation_qcodes


def load_human_qcode(file_path: str, is_test: bool = False):
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        human_qcodes: Set[str] = {l.rstrip("\n") for l in f.readlines()}
    return human_qcodes


def title_to_qcode(
    wiki_title: str,
    redirects: Dict[str, str],
    wikipedia_to_qcode: Dict[str, str],
    is_test: bool = False,
) -> Optional[str]:
    wiki_title = (
        wiki_title.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&le;", "≤")
        .replace("&ge;", "≥")
    )
    if len(wiki_title) == 0:
        return None
    wiki_title = wiki_title[0].upper() + wiki_title[1:]
    if wiki_title in redirects:
        wiki_title = redirects[wiki_title]
    if wiki_title in wikipedia_to_qcode:
        qcode = wikipedia_to_qcode[wiki_title]
        return qcode
    return None


def normalize_surface_form(surface_form: str, remove_the: bool = True):
    surface_form = surface_form.lower()
    surface_form = surface_form[4:] if surface_form[:4] == "the " and remove_the else surface_form
    return (
        unidecode(surface_form)
        .replace(".", "")
        .strip(" ")
        .replace('"', "")
        .replace("'s", "")
        .replace("'", "")
        .replace("`", "")
    )


def load_pem(pem_file: str, is_test: bool = False, max_cands: Optional[int] = None):
    pem: Dict[str, Any] = dict()
    line_num = 0
    with smart_open.smart_open(pem_file, "r") as f:
        if "s3" in pem_file:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=18e6, desc="Loading PEM"):
            line = ujson.loads(line)
            pem[line["surface_form"]] = line["qcode_probs"]
            if max_cands:
                pem[line["surface_form"]] = list(pem[line["surface_form"]].items())[:max_cands]
                # pem[line["surface_form"]] = line["qcode_probs"][:max_cands]
            line_num += 1
            if is_test and line_num > 10000:
                break
    return pem


def load_labels(file_path: str, is_test: bool = False, qcodes: Set[str] = None, keep_all_entities: bool = True):
    labels: Dict[str, str] = defaultdict(str)
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for i, line in tqdm(enumerate(f), total=80e6, desc="Loading Wikidata labels"):
            line = ujson.loads(line)
            if (qcodes is None or line["qcode"] in qcodes) or keep_all_entities:
                labels[line["qcode"]] = line["values"]
            if is_test and i > 10000:
                break
    return labels


def load_aliases(file_path: str, is_test: bool = False, qcodes: Set[str] = None, keep_all_entities: bool = True):
    aliases: Dict[str, List[str]] = defaultdict(list)
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for i, line in tqdm(enumerate(f), total=80e6, desc="Loading Wikidata aliases"):
            line = ujson.loads(line)
            if qcodes is None or line["qcode"] in qcodes or keep_all_entities:
                aliases[line["qcode"]] = line["values"]
            if is_test and i > 10000:
                break
    return aliases


def load_aida_means(file_path: str) -> Iterator[Tuple[str, str]]:
    with bz2.open(file_path, "rb") as f:
        for line in tqdm(f, total=18526177):
            line = line.decode("utf-8").rstrip("\n")
            surface_form, wiki_page = line.split("\t")
            surface_form = surface_form[1:-1]
            wiki_page = bytes(wiki_page.encode("utf-8")).decode("unicode-escape")
            yield surface_form, wiki_page


title_brackets_pattern = re.compile(" \(.*\)$")


def remove_wiki_brackets(title: str):
    m = title_brackets_pattern.search(title)
    if m is None:
        return title
    else:
        return title[: m.start()]


def load_wiki_to_tkid(file_path: str, is_test: bool = False):
    """
    Loads wiki title to evi tkid.
    s3://fount.resources.dev/evi_relations/wikipages_fixed_22.ujson/part-00000-ec31b93e-3036-46f2-9a6e-35ef5b458249-c000.ujson
    :param file_path: file path
    """
    en_wiki = "http://en.wikipedia.org/wiki/"
    wiki_to_tkid: Dict[str, str] = dict()
    i = 0
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=5e6, desc="Loading title to tkid"):
            i += 1
            line = ujson.loads(line)
            url = line["left"]
            if en_wiki in url:
                title = unquote(url[37:-3])
                tkid = line["right"]
                wiki_to_tkid[title] = tkid
            if is_test and i > 10000:
                break
    return wiki_to_tkid


def load_wikidata_to_tkid(file_path: str, is_test: bool = False):
    """
    Loads wiki title to evi tkid.
    s3://fount.resources.dev/evi_relations/wikidata_fixed_22.ujson/part-00000-ec31b93e-3036-46f2-9a6e-35ef5b458249-c000.ujson
    :param file_path: file path
    """
    qcode_to_tkid: Dict[str, str] = dict()
    i = 0
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        for line in tqdm(f, total=5e6, desc="Loading qcode to tkid"):
            i += 1
            line = ujson.loads(line)
            qcode = line["left"][2:-2].upper()
            tkid = line["right"]
            qcode_to_tkid[qcode] = tkid
            if is_test and i > 10000:
                break
    return qcode_to_tkid


def load_occuptations(file_path: str, is_test: bool = False):
    occuptations: Dict[str, Set[str]] = defaultdict(set)
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        i = 0
        for line in tqdm(f, total=4e6, desc="Loading Wikidata page occupations"):
            line = ujson.loads(line)
            occuptations[line["qcode"]] = set(line["values"])
            i += 1
            if is_test and i > 10000:
                break
    return occuptations


def load_sports(file_path: str, is_test: bool = False):
    sports: Dict[str, Set[str]] = defaultdict(set)
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        i = 0
        for line in tqdm(f, total=1e6, desc="Loading Wikidata page sports"):
            line = ujson.loads(line)
            sports[line["qcode"]] = set(line["values"])
            i += 1
            if is_test and i > 10000:
                break
    return sports


def load_country(file_path: str, is_test: bool = False):
    sports: Dict[str, Set[str]] = defaultdict(set)
    with smart_open.smart_open(file_path, "r") as f:
        if "s3" in file_path:
            f = io.StringIO(f.read())
        i = 0
        for line in tqdm(f, total=12e6, desc="Loading Wikidata page country"):
            line = ujson.loads(line)
            sports[line["qcode"]] = set(line["values"])
            i += 1
            if is_test and i > 10000:
                break
    return sports


def get_candidates(pem: Dict[str, Dict[str, float]], surface_form: str) -> Dict[str, float]:
    surface_form = normalize_surface_form(surface_form)
    if surface_form in pem:
        return pem[surface_form]
    return dict()


class ClassExplorer:
    def __init__(self, subclasses: Dict[str, List[str]]):
        self.subclasses: Dict[str, List[str]] = subclasses

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


def get_qcode_classes(
    qcode: str, occupations, sports, country, instance_of, class_explorer, subclasses
) -> Set[str]:
    classes = set()
    # subclasses disabled as types due to inconsistent usage in Wikidata
    # if qcode in subclasses_dict:
    #     classes.update({f'<subclass_of,{super_class}>' for super_class in subclasses_dict[qcode]})

    if qcode in occupations:
        # no brackets means subclasses will be used
        classes.update({super_class for super_class in occupations[qcode]})

    if qcode in sports:
        classes.update({f"<sports,{super_class}>" for super_class in sports[qcode]})

    if qcode in country:
        classes.update({f"<country,{super_class}>" for super_class in country[qcode]})

    if qcode in instance_of:
        classes.update(instance_of[qcode])

    # if qcode is a subclass then it implicitly is a class
    if qcode in subclasses:
        classes.add("Q16889133")

    return set(class_explorer.get_implied_classes(frozenset(classes)))
