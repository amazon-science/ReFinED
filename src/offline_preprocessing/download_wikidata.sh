#!/bin/bash
now=$(date +"%m_%d_%Y")
echo "Downloading latest-all.json.bz2"
curl -o "latest-all_$now.json.bz2" https://dumps.wikimedia.your.org/wikidatawiki/entities/latest-all.json.bz2
echo "Downloading latest-truthy_$now.nt.bz2"
curl -o "latest-truthy_$now.nt.bz2" https://dumps.wikimedia.your.org/wikidatawiki/entities/latest-truthy.nt.bz2
echo "Completed downloading Wikidata dump files"


