#!/bin/bash
now=$(date +"%m_%d_%Y")
echo "Downloading enwiki-latest-redirect_$now.sql.gz"
curl -o july_9_2020_enwiki-latest-redirect.sql.gz https://dumps.wikimedia.your.org/enwiki/latest/enwiki-latest-redirect.sql.gz
echo "Downloading enwiki-latest-page_$now.sql.gz"
curl -o july_9_2020_enwiki-latest-page.sql.gz https://dumps.wikimedia.your.org/enwiki/latest/enwiki-latest-page.sql.gz
echo "Downloading enwiki-latest-pages-articles_$now.xml.bz2"
curl -o july_9_2020_enwiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.your.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
echo "Completed Downloading Wikipedia dump files"