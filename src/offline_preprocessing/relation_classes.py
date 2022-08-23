import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Process cleaned Wikipedia, extract links, merge files.')
    parser.add_argument(
        "--wikidata_ekg_mappings_file",
        type=str,
        default='wikidata_to_ekg_mappings_sep.json',
        help="File path for cleaned wikipedia text with links extracted."
             "Location: s3://fount.resources.dev/distant_supervision_dataset/wikidata/wikidata_to_ekg_mappings.json."
    )
    parser.add_argument(
        "--query_value_constraint_file",
        type=str,
        default='query_value_constraint.json',
        help="File path for value constraint query output (SPARQL query commented below)"
    )
    parser.add_argument(
        "--query_type_constraint_file",
        type=str,
        default='query_type_constraint.json',
        help="File path for type constraint query output (SPARQL query commented below)"
    )
    parser.add_argument(
        "--current_classes",
        type=str,
        default='/big_data/wikidata/3_august_chosen_classes.txt',
        help="File path for type constraint query output (SPARQL query commented below)"
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
    args.output_dir = args.output_dir.rstrip('/')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.wikidata_ekg_mappings_file, 'r') as f:
        wikidata_props = {'P' + x for x in json.load(f).keys()}

    v_classes = get_all_classes(args.query_value_constraint_file, wikidata_props)
    s_classes = get_all_classes(args.query_type_constraint_file, wikidata_props)

    with open(args.current_classes, 'r') as f:
        current_classes = {cls.rstrip('\n') for cls in f.readlines()}

    print(f'Current classes length {len(current_classes)}')
    print(f'Value (object) classes length {len(v_classes)}')
    print(f'Type (subject) classes length {len(s_classes)}')
    print(f'Union (Value | Type) length {len(v_classes | s_classes)}')
    print(f'Union (Value | Type | Current) length {len(v_classes | s_classes | current_classes)}')
    with open(f'{args.output_dir}/current_classes_with_re.txt', 'w') as f:
        f.write('\n'.join([x for x in (v_classes | s_classes | current_classes)]))
    print(f'Written the Union (Value | Type | Current) to {args.output_dir}/current_classes_with_re.txt')


def get_all_classes(filename, wikidata_props):
    all_classes = set()
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            for item in line:
                if item['subj'].split('/')[-1] in wikidata_props:
                    classes = {cls.split('/')[-1] for cls in item['classes'].split(', ')}
                    all_classes.update(classes)
    return all_classes


if __name__ == '__main__':
    main()


# TODO query in code and download relevant resources in code
# SPARQL alternative
# SELECT ?subj (GROUP_CONCAT(DISTINCT ?class; SEPARATOR=", ") AS ?classes)
# WHERE
# {
#      ?subj p:P2302 ?statement.
#      ?statement ps:P2302 wd:Q21503250.  # replace with Q21510865 for value type constraints
#      ?statement pq:P2308 ?class.
#      ?statement pq:P2309 wd:Q21503252.
#      FILTER NOT EXISTS { ?subj wdt:P31 wd:Q19847637}
#      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
#
# }
# GROUP BY ?subj
