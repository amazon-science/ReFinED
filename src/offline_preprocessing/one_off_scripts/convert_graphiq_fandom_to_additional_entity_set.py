import json
from typing import List
from tqdm.auto import tqdm
import sys

sys.path.append('/home/ec2-user/data/refined_dates/code/src')
from offline_preprocessing.dataclasses_for_preprocessing import AdditionalEntity

additional_entities: List[AdditionalEntity] = []

with open('/home/ec2-user/data/refined_graphiq/data/fandom_entity_list.jsonl', 'r') as f:
    for line_num, line in tqdm(enumerate(f)):
        line = json.loads(line)
        label = line['name']
        aliases = line['aliases'] if line['aliases'] is not None else []
        if isinstance(aliases, str):
            aliases = [aliases]
        description = line['description']
        if label is None:
            continue
        if description is not None:
            description = description.replace(label, '')
            description = description.replace('is the', ' ')
            description = description.replace('is an', ' ')
            description = description.replace(' is ', ' ')
            description = description.replace(' was ', ' ')
            description = description.replace(' an ', ' ')
            description = description.replace(' a ', ' ')
            description = description.replace('.', '')
            description = description.replace('\'', '')
            description = description.replace('"', '')
            description = description.strip()
            description = description.replace('  ', ' ')
            description = description.replace('  ', ' ')
            description = description.replace('  ', ' ')

        universe = line['universe']
        if universe is not None:
            description = f'fictional character from {universe}'
            description = description.strip()

        if description is None:
            description = 'fictional character'

        entity_id = f'A{line_num}'

        additional_entities.append(
            AdditionalEntity(
                label=label,
                aliases=aliases,
                description=description,
                entity_types=['Q15632617'],
                entity_id=entity_id,
                graphiq_entity_id=line['unique_id']
            )
        )


with open('/home/ec2-user/data/refined_graphiq/data/fandom_entity_list_refined.jsonl2', 'w') as f_out:
    f_out.write('\n'.join([json.dumps(x.__dict__) for x in additional_entities]))
