import subprocess
import os
import argparse
import time
import datetime
import pathlib

this_file_dir = pathlib.Path(__file__).parent.absolute()


def str2bool(v):
    """
    Helper function for allowing passing of boolean command line arguments
    """
    if isinstance(v, bool):
        return v
    if str(v).lower() in {'true', 'y', 't'}:
        return True
    elif str(v).lower() in {'false', 'n', 'f'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def preprocess_wikipedia(dump_path, save_path, keep_links=True, remove_disambig_pages=True, keep_categories=True,
                         force_overwrite=False):
    """
    Initiate wikipedia preprocessing (cleaning of text, removing tables etc.) using wiki_extractor.py
    Cleaned dump will be around 18GB
    """

    # Create directory to dump articles
    if os.path.exists(save_path):
        if not force_overwrite:
            raise Exception('Dump of cleaned articles already exists - run with force_overwrite=True if wish to '
                            'overwrite')
    else:
        os.makedirs(save_path, exist_ok=True)

    cmd = f'python3 {this_file_dir}/wiki_extractor.py {dump_path} -o {save_path} --json -b 50m'

    if keep_links:
        cmd += ' --links'

    if remove_disambig_pages:
        cmd += ' --filter_disambig_pages'

    if keep_categories:
        cmd += ' --extract_categories'

    st = time.time()

    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    print(output, error)

    total_time = str(datetime.timedelta(seconds=time.time() - st))

    print('Finished. Total time:', total_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_path', default='./data/enwiki-latest-pages-articles.xml.bz2',
                        help='Path to wikipedia dump')
    parser.add_argument('--save_dir', default='./data/wikipedia/wiki_cleaned/',
                        help='Path to save cleaned wikipedia dump')
    parser.add_argument('--keep_links', default=True, type=str2bool, help='Whether to keep hyperlink information')
    parser.add_argument('--remove_disambig_pages', default=True, type=str2bool, help='Whether to remove disambiguation'
                                                                                     'pages')
    parser.add_argument('--keep_categories', default=True, type=str2bool, help='Whether to keep the categories for '
                                                                               'each page')
    parser.add_argument('--force_overwrite', default=False, type=str2bool, help='If True, overwrite existing cleaned '
                                                                                'dump if one exists')
    args = parser.parse_args()

    preprocess_wikipedia(args.dump_path, args.save_dir, args.keep_links, args.remove_disambig_pages,
                         args.keep_categories, args.force_overwrite)
