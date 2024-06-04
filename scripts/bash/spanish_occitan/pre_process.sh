#!/bin/bash -v

data_dir=./data/opus/spanish_occitan
data_output=./data/opus/preprocess/spanish_occitan
fast_text=./tools/fast_text/fastText
moses_decoder=./tools/mosesdecoder/scripts

## preprocess the data before
## LUCIA: add a pipe with different preprocessing (maybe a truecase?)
for file in CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu WikiMatrix XLEnt wikimedia
  do
    cat ${data_dir}/${file}.es-oc.es |
    sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g' | sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
    ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es > ${data_output}/${file}.es-oc.es.prepros
    ### TODO: select LANGUAGE
    cat ${data_dir}/${file}.es-oc.oc |
    sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g'| sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
    ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es > ${data_output}/${file}.es-oc.oc.prepros
done

## language calssification using fastext
for file in CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu WikiMatrix XLEnt wikimedia
  do
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.es-oc.es.prepros > ${data_output}/${file}.es-oc.es.lang_classification.fast_text
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.es-oc.oc.prepros > ${data_output}/${file}.es-oc.oc.lang_classification.fast_text

    paste ${data_output}/${file}.es-oc.es.prepros ${data_output}/${file}.es-oc.oc.prepros ${data_output}/${file}.es-oc.es.lang_classification.fast_text \
    | awk -F'\t' '{if ($3=="__label__es") {print $1 > "'${data_output}'/'${file}'.es-oc.es.filtered"; print $2 > "'${data_output}'/'${file}'.es-oc.oc.filtered"}}'
done

## add the ratio

for file in CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu WikiMatrix XLEnt wikimedia
do
  ## language calssification using fastext
  echo "stats for: " ${file}
  echo "#sents"
  wc -l ${data_output}/${file}.es-oc.es.prepros
  echo "#words"
  wc -w ${data_output}/${file}.es-oc.es.prepros
  wc -w ${data_output}/${file}.es-oc.oc.prepros
  echo "#vocab"
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.es-oc.es.prepros | sort | uniq -c | wc -l
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.es-oc.oc.prepros | sort | uniq -c | wc -l

  echo "filtered: " ${file}
  echo "#sents"
  wc -l ${data_output}/${file}.es-oc.es.filtered
  echo "#words"
  wc -w ${data_output}/${file}.es-oc.es.filtered
  wc -w ${data_output}/${file}.es-oc.oc.filtered
  echo "#vocab"
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.es-oc.es.filtered | sort | uniq -c | wc -l
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.es-oc.oc.filtered | sort | uniq -c | wc -l
done
