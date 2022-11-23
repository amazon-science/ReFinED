from typing import Set

import requests

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
