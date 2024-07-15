#!/bin/bash -v

data_dir=./data/opus/spanish_occitan
data_tmp=./data/opus/preprocess/spanish_occitan
data_output=./experiments/spanish_occitan/data
fast_text=./tools/fast_text/fastText
moses_decoder=./tools/mosesdecoder/scripts
python_scripts=./scripts/python
src=es
tgt=oc
pair=es-oc

#FILES=(CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu WikiMatrix XLEnt wikimedia)
FILES=(GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu WikiMatrix XLEnt wikimedia)

rm -r ${data_tmp}
mkdir -p ${data_output}
mkdir -p ${data_tmp}

rm ${data_output}/train.*


for file in ${FILES[@]}
  do
    ${moses_decoder}/training/clean-corpus-n.perl ${data_dir}/${file}.${pair} ${src} ${tgt} ${data_tmp}/${file}.original.clean 2 60
    ##
    cat ${data_tmp}/${file}.original.clean.${src} >> ${data_output}/train.original.${src}
    cat ${data_tmp}/${file}.original.clean.${tgt} >> ${data_output}/train.original.${tgt} 
done

rm -r {data_tmp}



### OLD CODE 
## preprocess the data before
#for file in ${FILES[@]}
#  do
#  cat ${data_dir}/${file}.${pair}.${src} |
#    ${moses_decoder}/tokenizer/normalize-punctuation.perl -l ${src} |
#    ${moses_decoder}/tokenizer/tokenizer.perl -l ${src} > ${data_tmp}/${file}.${pair}.prepros.${src}
#  cat ${data_dir}/${file}.${pair}.${tgt} |
#    ${moses_decoder}/tokenizer/normalize-punctuation.perl -l ${src} |
#    ${moses_decoder}/tokenizer/tokenizer.perl -l ${src} > ${data_tmp}/${file}.${pair}.prepros.${tgt}
#
#    ${moses_decoder}/training/clean-corpus-n.perl ${data_tmp}/${file}.${pair}.prepros ${src} ${tgt} ${data_tmp}/${file}.clean 1 120 
#  
#done
#
### language calssification using fastext
#for file in ${FILES[@]} 
#  do
#    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_tmp}/${file}.clean.${src} > ${data_tmp}/${file}.${src}.lang_classification.fast_text
#    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_tmp}/${file}.clean.${tgt} > ${data_tmp}/${file}.${tgt}.lang_classification.fast_text
#
#    paste ${data_tmp}/${file}.clean.${src} ${data_tmp}/${file}.clean.${tgt} ${data_tmp}/${file}.${src}.lang_classification.fast_text \
#    | awk -F'\t' '{if ($3=="__label__es") {print $1 > "'${data_tmp}'/'${file}'.filtered.'${src}'"; print $2 > "'${data_tmp}'/'${file}'.filtered.'${tgt}'"}}'
#done
#
### add the ratio
#for file in ${FILES[@]}
#do
#    python3 ${python_scripts}/preprocess/lev_ratio.py ${data_tmp}/${file}.filtered.${tgt} ${data_tmp}/${file}.filtered.${src} ${data_tmp}/${file}.levenshtein.${src} ${data_tmp}/${file}.levenshtein.${tgt}
#done
#
#
#echo "" > ./experiments/data/train.{${src},${tgt}}
#echo "" > ./experiments/data/train.{original,filtered,levenshtein}.{${src},${tgt}}
#for file in ${FILES[@]} 
#do
#  cat ${data_tmp}/${file}.clean.${src} >> ${data_output}/train.original.${src}
#  cat ${data_tmp}/${file}.clean.${tgt} >> ${data_output}/train.original.${tgt}
#  ###
#  cat ${data_tmp}/${file}.filtered.${src} >> ${data_output}/train.filtered.${src}
#  cat ${data_tmp}/${file}.filtered.${tgt} >> ${data_output}/train.filtered.${tgt}
#  ###   
#  cat ${data_tmp}/${file}.levenshtein.${src} >> ${data_output}/train.levensthein.${src}
#  cat ${data_tmp}/${file}.levenshtein.${tgt} >> ${data_output}/train.levenshtein.${tgt}
#done

