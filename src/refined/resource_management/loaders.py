import bz2
import io
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import unquote

import ujson
from tqdm.auto import tqdm
from unidecode import unidecode

from refined.utilities.general_utils import get_logger

LOG = get_logger(__name__)


def load_qcode_to_idx(filename: str, is_test: bool = False) -> Dict[str, int]:
    LOG.info("Loading qcode_to_idx")
    qcode_to_idx: Dict[str, int] = dict()
    line_num = 0
    with open(filename, "r") as f:
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
    with open(filepath, "r") as f:
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
    with open(file_path, "r") as f:
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
    with open(file_path, "r") as f:
        for line in tqdm(f, total=2e6, desc="Loading Wikidata subclasses"):
            line = ujson.loads(line.replace("\x00",""))
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
    with open(file_path, "r") as f:
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
    with open(file_path, "r") as f:
        for line in tqdm(f, total=8e6, desc="Loading enwiki sitelinks"):
            line = ujson.loads(line)
            enwiki_to_qcode[line["values"].replace(" ", "_")] = line["qcode"]
            line_num += 1
            if is_test and line_num > 10000:
                break
    LOG.info(f"Loaded enwiki_to_qcode, size = {len(enwiki_to_qcode)}")
    return enwiki_to_qcode


def load_disambiguation_qcodes(file_path: str, is_test: bool = False):
    with open(file_path, "r") as f:
        disambiguation_qcodes: Set[str] = {l.rstrip("\n") for l in f.readlines()}
    return disambiguation_qcodes


def load_human_qcode(file_path: str, is_test: bool = False):
    with open(file_path, "r") as f:
        human_qcodes: Set[str] = {l.rstrip("\n") for l in f.readlines()}
    return human_qcodes


def normalize_surface_form(surface_form: str, remove_the: bool = True):
    surface_form = surface_form.lower()
    surface_form = surface_form[:-1] if surface_form.endswith(',') else surface_form
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
    with open(pem_file, "r") as f:
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
    with open(file_path, "r") as f:
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
    with open(file_path, "r") as f:
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

def load_occuptations(file_path: str, is_test: bool = False):
    occuptations: Dict[str, Set[str]] = defaultdict(set)
    with open(file_path, "r") as f:
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
    with open(file_path, "r") as f:
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
    with open(file_path, "r") as f:
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

    return set(class_explorer._get_implied_classes(frozenset(classes)))


def get_candidates(pem: Dict[str, Dict[str, float]], surface_form: str) -> Dict[str, float]:
    surface_form = normalize_surface_form(surface_form)
    if surface_form in pem:
        return pem[surface_form]
    return dict()
