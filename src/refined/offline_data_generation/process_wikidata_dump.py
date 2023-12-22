import bz2
import json

from tqdm.auto import tqdm
import argparse
import os
from types import SimpleNamespace
from refined.utilities.general_utils import get_logger

LOG = get_logger(__name__)

def extract_useful_info(entity,languages):
    qcode = entity['id']
    all_data = []
    for language in languages: # english is the main language for the description. But the label can be multiple languages. 
        if language in entity['labels']:
            entity_en_label = entity['labels'][language]['value']   
        elif 'en' in entity['labels']: 
            entity_en_label = entity['labels']['en']['value'] 
        else:
            entity_en_label = None
           
        if 'en' in entity['descriptions']:
            entity_en_desc = entity['descriptions']['en']['value']
        elif language in entity['descriptions']:
            entity_en_desc = entity['descriptions'][language]['value']
        else:
            entity_en_desc = None

        if language in entity['aliases']:
            entity_en_aliases = [alias['value'] for alias in entity['aliases'][language]]
        else:
            entity_en_aliases = []

        if 'sitelinks' in entity:
            sitelinks = entity['sitelinks']
        else:
            sitelinks = {}

        if f'{language}wiki' in sitelinks:
            enwiki_title = sitelinks[f'{language}wiki']['title']
        else:
            enwiki_title = None


        sitelinks_cnt = len(sitelinks.items())
        statements_cnt = 0
        triples = {}
        for pcode, objs in entity['claims'].items():
            # group by pcode -> [list of qcodes]
            for obj in objs:
                statements_cnt += 1
                if not obj['mainsnak']['datatype'] == 'wikibase-item' or obj['mainsnak']['snaktype'] == 'somevalue' \
                        or 'datavalue' not in obj['mainsnak']:
                    continue
                if pcode not in triples:
                    triples[pcode] = []
                triples[pcode].append(obj['mainsnak']['datavalue']['value']['id'])
                
        all_data.append({'qcode': qcode, 'label': entity_en_label, 'desc': entity_en_desc,
                'aliases': entity_en_aliases, 'sitelinks_cnt': sitelinks_cnt, 'enwiki': enwiki_title,
                'statements_cnt': statements_cnt, 'triples': triples})  
    
    return all_data


def build_wikidata_lookups(languages,args_override=None):
    if args_override is None:
        parser = argparse.ArgumentParser(description='Build lookup dictionaries from Wikidata JSON dump.')
        parser.add_argument(
            "--dump_file_path",
            default='latest-all.json.bz2',
            type=str,
            help="file path to JSON Wikidata dump file (latest-all.json.bz2)"
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default='output',
            help="Directory where the lookups will be stored"
        )
        parser.add_argument(
            "--overwrite_output_dir",
            action="store_true",
            help="Overwrite the content of the output directory"
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="mode for testing (only processes first 500 lines)"
        )
        args = parser.parse_args()
    else:
        args = SimpleNamespace(**args_override)
    args.output_dir = args.output_dir.rstrip('/')
    number_lines_to_process = 500 if args.test else 1e20
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    i = 0

    # duplicate names in output_files to make removing .part easier after pre-processing has finished.
    filenames = [
        f'{args.output_dir}/sitelinks_cnt.json.part',
        f'{args.output_dir}/statements_cnt.json.part',
        f'{args.output_dir}/enwiki.json.part',
        f'{args.output_dir}/desc.json.part',
        f'{args.output_dir}/aliases.json.part',
        f'{args.output_dir}/qcode_to_label.json.part',
        f'{args.output_dir}/instance_of_p31.json.part',
        f'{args.output_dir}/country_p17.json.part',
        f'{args.output_dir}/sport_p641.json.part',
        f'{args.output_dir}/occupation_p106.json.part',
        f'{args.output_dir}/subclass_p279.json.part',
        f'{args.output_dir}/pcodes.json.part',
        f'{args.output_dir}/human_qcodes.json.part',
        f'{args.output_dir}/disambiguation_qcodes.txt.part',
        f'{args.output_dir}/triples.json.part',
        f'{args.output_dir}/located_in_p131.json.part',
    ]

    output_files = {
        'sitelinks_cnt': open(f'{args.output_dir}/sitelinks_cnt.json.part', 'w'),
        'statements_cnt': open(f'{args.output_dir}/statements_cnt.json.part', 'w'),
        'enwiki': open(f'{args.output_dir}/enwiki.json.part', 'w'),
        'desc': open(f'{args.output_dir}/desc.json.part', 'w'),
        'aliases': open(f'{args.output_dir}/aliases.json.part', 'w'),
        'label': open(f'{args.output_dir}/qcode_to_label.json.part', 'w'),
        'instance_of_p31': open(f'{args.output_dir}/instance_of_p31.json.part', 'w'),
        'country_p17': open(f'{args.output_dir}/country_p17.json.part', 'w'),
        'sport_p641': open(f'{args.output_dir}/sport_p641.json.part', 'w'),
        'occupation_p106': open(f'{args.output_dir}/occupation_p106.json.part', 'w'),
        'subclass_p279': open(f'{args.output_dir}/subclass_p279.json.part', 'w'),
        'properties': open(f'{args.output_dir}/pcodes.json.part', 'w'),
        'humans': open(f'{args.output_dir}/human_qcodes.json.part', 'w'),
        'disambiguation': open(f'{args.output_dir}/disambiguation_qcodes.txt.part', 'w'),
        'triples': open(f'{args.output_dir}/triples.json.part', 'w'),
        'located_in_p131': open(f'{args.output_dir}/located_in_p131.json.part', 'w'),
    }

    with bz2.open(args.dump_file_path, 'rb') as f:
        for line in tqdm(f):
            i += 1
            if len(line) < 3:
                continue
            line = line.decode('utf-8').rstrip(',\n')
            line = json.loads(line)
            entity_contents = extract_useful_info(line,languages)
#             LOG.info(entity_contents)
#             raise Exception(entity_contents)
            for entity_content in entity_contents:
                qcode = entity_content['qcode']
                if 'P' in qcode:
                    output_files['properties'].write(json.dumps({'qcode': qcode, 'values': entity_content}) + '\n')

                if entity_content['sitelinks_cnt']:
                    output_files['sitelinks_cnt'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['sitelinks_cnt']}) + '\n')

                if entity_content['statements_cnt']:
                    output_files['statements_cnt'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['statements_cnt']}) + '\n')

                if entity_content['enwiki']:
                    output_files['enwiki'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['enwiki']}) + '\n')

                if entity_content['desc']:
                    output_files['desc'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['desc']}) + '\n')

                if entity_content['aliases']:
                    output_files['aliases'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['aliases']}) + '\n')

                if entity_content['label']:
                    output_files['label'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['label']}) + '\n')

                # <instance of, class> We use <instance of, class> as entity types (one entity can be multiple types).
                if 'P31' in entity_content['triples']:
                    output_files['instance_of_p31'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P31']}) + '\n')
                    # <instance of, class>
                    if 'Q5' in entity_content['triples']['P31'] or 'Q15632617' in entity_content['triples']['P31']:
                        output_files['humans'].write(str(qcode) + '\n')
                    # <instance of, class>
                    if 'Q4167410' in entity_content['triples']['P31'] or 'Q22808320' in entity_content['triples']['P31']:
                        output_files['disambiguation'].write(str(qcode) + '\n')
                
                # <instance of, class>
                if 'P131' in entity_content['triples']:
                    output_files['located_in_p131'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P131']}) + '\n')

                # <country, class>
                if 'P17' in entity_content['triples']:
                    output_files['country_p17'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P17']}) + '\n')

                # <sport, class>
                if 'P641' in entity_content['triples']:
                    output_files['sport_p641'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P641']}) + '\n')

                # <occupation, class>
                if 'P106' in entity_content['triples']:
                    output_files['occupation_p106'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P106']}) + '\n')

                # <subclass, class>
                if 'P279' in entity_content['triples']:
                    output_files['subclass_p279'] \
                        .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P279']}) + '\n')

                output_files['triples'].write(json.dumps({'qcode': qcode, 'triples': entity_content['triples']}) + '\n')

            if i > number_lines_to_process:
                break

    for file in output_files.values():
        file.close()

    if not args.test:
        for filename in filenames:
            os.rename(filename, filename.replace('.part', ''))


if __name__ == '__main__':
    build_wikidata_lookups()
