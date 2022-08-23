import json
from tqdm.auto import tqdm

total_clean_hyperlinks = 0
total_hyperlinks = 0

qcodes = set()
with open('wikipedia_links_aligned.json', 'r') as f:
    for line_num, line in tqdm(enumerate(f), total=6000000):
        line = json.loads(line)
        total_clean_hyperlinks += len(line['hyperlinks_clean'])
        total_hyperlinks += len(line['hyperlinks'])
        qcodes.update({x['qcode'] for x in line['hyperlinks_clean']})
        if line_num % 100000 == 1:
            print('*'*25)
            print('Line number:', line_num)
            print(f'Number of hyperlinks {total_hyperlinks:,}')
            print(f'Number of hyperlinks mapped to Wikidata {total_clean_hyperlinks:,}')
            print(f'Percentage of hyperlinks mapped to Wikidata {total_clean_hyperlinks / total_hyperlinks * 100:.1f}%')
            print(f'Number of unique qcodes {len(qcodes):,}')
            print(f'Average number of entities per page {total_clean_hyperlinks / line_num:.1f}')
            print('*'*25)


print('*' * 25)
print('Final:', line_num)
print(f'Number of hyperlinks {total_clean_hyperlinks:,}')
print(f'Number of hyperlinks mapped to Wikidata {total_hyperlinks:,}')
print(f'Percentage of hyperlinks mapped to Wikidata {total_clean_hyperlinks / total_hyperlinks * 100:.1f}%')
print(f'Number of unique qcodes {len(qcodes):,}')
print(f'Average number of entities per page {total_clean_hyperlinks / line_num:.1f}')
print('*' * 25)


# Line number: 6000001
# Number of hyperlinks 101,934,606
# Number of hyperlinks mapped to Wikidata 105,895,699
# Percentage of hyperlinks mapped to Wikidata 96.3%
# Number of unique qcodes 3,894,401
# Average number of entities per page 17.0
