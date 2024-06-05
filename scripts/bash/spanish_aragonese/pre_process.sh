#!/bin/bash -v

data_dir=./data/opus/spanish_aragonese
data_output=./data/opus/preprocess/spanish_aragonese
fast_text=./tools/fast_text/fastText
moses_decoder=./tools/mosesdecoder/scripts

## preprocess the data before
for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
  cat ${data_dir}/${file}.an-es.es |
      sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g' | sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
      perl ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es > ${data_output}/${file}.an-es.es.prepros
  ### TODO: select LANGUAGE
  cat ${data_dir}/${file}.an-es.an |
      sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g' | sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
      perl ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es > ${data_output}/${file}.an-es.an.prepros

  ##### to build stats
done

## language calssification using fastext
for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.an-es.es.prepros > ${data_output}/${file}.an-es.es.lang_classification.fast_text
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.an-es.an.prepros > ${data_output}/${file}.an-es.an.lang_classification.fast_text

    echo "" > ${data_output}/${file}.an-es.es.filtered
    echo "" > ${data_output}/${file}.an-es.an.filtered
    paste ${data_output}/${file}.an-es.es.prepros ${data_output}/${file}.an-es.an.prepros ${data_output}/${file}.an-es.es.lang_classification.fast_text \
    | awk -F'\t' '{if ($3=="__label__es") {print $1 > "'${data_output}'/'${file}'.an-es.es.filtered"; print $2 > "'${data_output}'/'${file}'.an-es.an.filtered"}}'
done

## add the ratio
for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
	python3 ${python_scripts}/preprocess/lev_ratio.py ${data_output}/${file}.an-es.an.filtered ${data_output}/${file}.an-es.es.filtered
done

## stats for each corpus
for file in GNOME QED Tatoeba Ubuntu WikiMatrix XLent wikimedia
do
  ## language calssification using fastext
  echo "stats for: " ${file}
  echo "#sents"
  wc -l ${data_output}/${file}.an-es.es.prepros
  echo "#words"
  wc -w ${data_output}/${file}.an-es.es.prepros
  wc -w ${data_output}/${file}.an-es.an.prepros
  echo "#vocab"
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.es.prepros | sort | uniq -c | wc -l
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.an.prepros | sort | uniq -c | wc -l

  echo "filtered: " ${file}
  echo "#sents"
  wc -l ${data_output}/${file}.an-es.es.filtered
  echo "#words"
  wc -w ${data_output}/${file}.an-es.es.filtered
  wc -w ${data_output}/${file}.an-es.an.filtered
  echo "#vocab"
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.es.filtered | sort | uniq -c | wc -l
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.an-es.an.filtered | sort | uniq -c | wc -l
done
