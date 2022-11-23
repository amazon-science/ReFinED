from typing import Optional, Set, Mapping

from refined.resource_management.lmdb_wrapper import LmdbImmutableDict
from refined.resource_management.loaders import load_disambiguation_qcodes
from refined.resource_management.resource_manager import ResourceManager


class WikidataMapper:
    def __init__(self, resource_manager: ResourceManager):
        resources_to_files = resource_manager.get_additional_data_files()
        self.redirects: Mapping[str, str] = LmdbImmutableDict(resources_to_files["redirects"])
        self.wiki_to_qcode: Mapping[str, str] = LmdbImmutableDict(resources_to_files["wiki_to_qcode"])
        self.disambiguation_qcodes: Set[str] = load_disambiguation_qcodes(resources_to_files["disambiguation_qcodes"])
        self.qcode_to_label: Mapping[str, str] = LmdbImmutableDict(resources_to_files["qcode_to_label"])

    def map_title_to_wikidata_qcode(self, wiki_title: str) -> Optional[str]:
        wiki_title = (
            wiki_title.replace("&lt;", "<").replace("&gt;", ">").replace("&le;", "≤").replace("&ge;", "≥")
        )
        if len(wiki_title) == 0:
            return None
        wiki_title = wiki_title[0].upper() + wiki_title[1:]
        if wiki_title in self.redirects:
            wiki_title = self.redirects[wiki_title]
        if wiki_title in self.wiki_to_qcode:
            qcode = self.wiki_to_qcode[wiki_title]
            return qcode
        return None

    def wikidata_qcode_is_disambiguation_page(self, qcode: str) -> bool:
        return qcode in self.disambiguation_qcodes
