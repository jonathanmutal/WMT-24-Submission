#!/bin/bash -v

data_dir=./data/opus/spanish_aragonese
data_output=./data/opus/preprocess/spanish_aragonese
fast_text=./tools/fast_text/fastText
moses_decoder=./tools/mosesdecoder/scripts
python_scripts=./scripts/python

## preprocess the data before
for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
  cat ${data_dir}/${file}.an-es.es |
      sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g' | sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
      perl ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es > ${data_output}/${file}.an-es.prepros.es
  cat ${data_dir}/${file}.an-es.an |
      sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g' | sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
      perl ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es > ${data_output}/${file}.an-es.prepros.an

     ${moses_decoder}/training/clean-corpus.perl ${data_output}/${file}.prepros.an-es es an ${data_output}/${file}.clean 1 120 
done

## language calssification using fastext
for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.clean.es > ${data_output}/${file}.an-es.es.lang_classification.fast_text
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.clean.an > ${data_output}/${file}.an-es.an.lang_classification.fast_text

    echo "" > ${data_output}/${file}.an-es.es.filtered
    echo "" > ${data_output}/${file}.an-es.an.filtered
    paste ${data_output}/${file}.es.clean ${data_output}/${file}.an.clean ${data_output}/${file}.an-es.es.lang_classification.fast_text \
    | awk -F'\t' '{if ($3=="__label__es") {print $1 > "'${data_output}'/'${file}'.es.filtered"; print $2 > "'${data_output}'/'${file}'.an.filtered"}}'
done

## add the ratio
for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
    python3 ${python_scripts}/preprocess/lev_ratio.py ${data_output}/${file}.an.filtered ${data_output}/${file}.es.filtered ${data_output}/${file}.es.levensthein ${data_output}/${file}.an.levensthein
done

for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
  cat ${data_output}/${file}.clean.es >> ./experiments/data/original/train.es
  cat ${data_output}/${file}.clean.an >> ./experiments/data/original/train.an
  ###
  cat ${data_output}/${file}.filtered.es >> ./experiments/data/filter/train.es
  cat ${data_output}/${file}.filtered.an >> ./experiments/data/filter/train.an
  ###   
  cat ${data_output}/${file}.es.levenshtein >> ./experiments/data/filter.radio/train.es
  cat ${data_output}/${file}.an.levenshtein >> ./experiments/data/filter.radio/train.an
done
## stats for each corpus
#for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
#do
#  ## language calssification using fastext
#  echo "stats for: " ${file}
#  echo "#sents"
#  wc -l ${data_output}/${file}.an-es.es.prepros
#  echo "#words"
#  wc -w ${data_output}/${file}.an-es.es.prepros
#  wc -w ${data_output}/${file}.an-es.an.prepros
#  echo "#vocab"
#  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.es.prepros | sort | uniq -c | wc -l
#  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.an.prepros | sort | uniq -c | wc -l
#
#  echo "filtered: " ${file}
#  echo "#sents"
#  wc -l ${data_output}/${file}.an-es.es.filtered
#  echo "#words"
#  wc -w ${data_output}/${file}.an-es.es.filtered
#  wc -w ${data_output}/${file}.an-es.an.filtered
#  echo "#vocab"
#  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.es.filtered | sort | uniq -c | wc -l
#  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.an.filtered | sort | uniq -c | wc -l
#done
