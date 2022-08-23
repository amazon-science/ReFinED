import ujson as json
from tqdm.auto import tqdm

with open('geography_subgraph_only.json', 'w') as f_out:
    with open('geography_subgraph.json', 'r') as f:
        for line in tqdm(f, total=1000000):
            line = json.loads(line)
            qcode = line['qcode']
            triples = line['triples']
            if 'P131' in triples:
                located_in = triples['P131']
                f_out.write(json.dumps({'qcode': qcode, 'values': located_in}) + '\n')
            break
