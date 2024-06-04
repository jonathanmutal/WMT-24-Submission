#!/bin/bash -v

data_dir=./data/opus/spanish_asturian
data_output=./data/opus/preprocess/spanish_asturian
fast_text=./tools/fast_text/fastText
moses_decoder=./tools/mosesdecoder/scripts

## preprocess the data before
## LUCIA: add a pipe with different preprocessing (maybe a truecase?)
for file in CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu wikimedia
  do
    cat ${data_dir}/${file}.ast-es.es |
    sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g' | sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
    perl ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es  > ${data_output}/${file}.ast-es.es.prepros
    ### TODO: select LANGUAGE
    cat ${data_dir}/${file}.ast-es.ast |
    sed -r 's/^ //g' | sed -r 's/\\t//g' | sed -r 's/\_//g' | sed -r 's/« /\"/g' | sed -r 's/ »/\"/g' |
    perl ${moses_decoder}/tokenizer/normalize-punctuation.perl -l es  > ${data_output}/${file}.ast-es.ast.prepros
done

## language calssification using fastext
for file in CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu wikimedia
  do
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.ast-es.es.prepros > ${data_output}/${file}.ast-es.es.lang_classification.fast_text
    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_output}/${file}.ast-es.ast.prepros > ${data_output}/${file}.ast-es.ast.lang_classification.fast_text

    paste ${data_output}/${file}.ast-es.es.prepros ${data_output}/${file}.ast-es.ast.prepros ${data_output}/${file}.ast-es.es.lang_classification.fast_text \
    | awk -F'\t' '{if ($3=="__label__es") {print $1 > "'${data_output}'/'${file}'.ast-es.es.filtered"; print $2 > "'${data_output}'/'${file}'.ast-es.ast.filtered"}}'
done

## add the ratio


for file in CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu wikimedia
do
  ## language calssification using fastext
  echo "stats for: " ${file}
  echo "#sents"
  wc -l ${data_output}/${file}.ast-es.es.prepros
  echo "#words"
  wc -w ${data_output}/${file}.ast-es.es.prepros
  wc -w ${data_output}/${file}.ast-es.ast.prepros
  echo "#vocab"
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.ast-es.es.prepros | sort | uniq -c | wc -l
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.ast-es.ast.prepros | sort | uniq -c | wc -l

  echo "filtered: " ${file}
  echo "#sents"
  wc -l ${data_output}/${file}.ast-es.es.filtered
  echo "#words"
  wc -w ${data_output}/${file}.ast-es.es.filtered
  wc -w ${data_output}/${file}.ast-es.ast.filtered
  echo "#vocab"
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.ast-es.es.filtered | sort | uniq -c | wc -l
  tr -cs '[:alnum:]' '[\n*]' < ${data_output}/${file}.ast-es.ast.filtered | sort | uniq -c | wc -l
done
