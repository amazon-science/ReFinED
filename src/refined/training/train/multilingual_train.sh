#!/bin/sh
# export NCCL_SOCKET_IFNAME=<VALUE>
export NCCL_IB_DISABLE=1

export PYTHONPATH=$PYTHONPATH:src
temperature_scaling=0.02
transformer_name='bert-base-multilingual-cased' # bert-base-multilingual-cased, roberta-base
model_name='ReFinED_mBERT_11_languages_temp'${temperature_scaling}
languages='ar_de_en_es_fa_ja_sr_ta_tr_fr_it'

python3 src/refined/training/train/train.py --experiment_name ${model_name} --transformer_name ${transformer_name} --data data_combine_11_languages_wikidata_all_eng_label_desc --entity_set wikidata --languages ${languages} --temperature_scaling ${temperature_scaling} > fine_tuned_model/${model_name}_training_output.txt