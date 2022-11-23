import os
import urllib.request
from typing import Set

import requests
from tqdm import tqdm

DENY_CLASSES = {'Q13406463', 'Q4167410', 'Q101352', 'Q21528878', 'Q22808320', 'Q12308941', 'Q11879590',
                'Q11879590', 'Q12308941', 'Q4167836', 'Q21528878', 'Q22808320', 'Q27924673',
                'Q22808320', 'Q25052136', 'Q48522', 'Q15407973',

                # filter frequent entities in Wikidata that rarely appear in text
                'Q13442814', 'Q318', 'Q523', 'Q11173', 'Q1931185', 'Q30612', 'Q67201574', 'Q497654', 'Q67015883'
                }

# deny_classes = {'Q13442814', 'Q318', 'Q523', 'Q11173', 'Q1931185', 'Q30612', 'Q67201574', 'Q497654'}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url_with_progress_bar(url: str, output_path: str, redownload: bool = False):
    if not os.path.exists(output_path) or redownload:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path + '.part', reporthook=t.update_to)
            os.rename(output_path + '.part', output_path)
    else:
        print(f'{output_path} already exists so skipping the download.')


wikidata_sparql_endpoint = 'https://query.wikidata.org/sparql'
query_relation_obj_classes = """
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

query_relation_subj_classes = """
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
    object_classes_res = requests.get(wikidata_sparql_endpoint, params={'format': 'json',
                                                                        'query': query_relation_obj_classes}).json()
    subject_classes_res = requests.get(wikidata_sparql_endpoint, params={'format': 'json',
                                                                         'query': query_relation_subj_classes}).json()
    obj_classes = {x['class']['value'][31:] for x in object_classes_res['results']['bindings']}
    subj_classes = {x['class']['value'][31:] for x in subject_classes_res['results']['bindings']}
    return obj_classes | subj_classes
