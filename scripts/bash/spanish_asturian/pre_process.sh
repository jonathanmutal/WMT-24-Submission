#!/bin/bash -v
data_dir=./data/opus/spanish_asturian
data_tmp=./data/opus/preprocess/spanish_asturian
data_output=./experiments/spanish_asturian/data
fast_text=./tools/fast_text/fastText
moses_decoder=./tools/mosesdecoder/scripts
python_scripts=./scripts/python
src=es
tgt=ast
pair=ast-es

rm -r ${data_tmp}
mkdir -p ${data_output}
mkdir -p ${data_tmp}

rm ${data_output}/train.*{lev,original}*

#FILES=(CCMatrix GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu wikimedia)
FILES=(GNOME KDE4 NLLB QED TED2020 Tatoeba Ubuntu wikimedia XLEnt)

# do nothing but clean the long lines for llm
for file in ${FILES[@]}
  do
    ${moses_decoder}/training/clean-corpus-n.perl ${data_dir}/${file}.${pair} ${src} ${tgt} ${data_tmp}/${file}.original.clean 1 60
    ##
    cat ${data_tmp}/${file}.original.clean.${src} >> ${data_output}/train.original.${src}
    cat ${data_tmp}/${file}.original.clean.${tgt} >> ${data_output}/train.original.${tgt} 
done


## preprocess the data before
for file in ${FILES[@]}
  do
  cat ${data_tmp}/${file}.original.clean.${src} |
    ${moses_decoder}/tokenizer/normalize-punctuation.perl -l ${src} |
    ${moses_decoder}/tokenizer/tokenizer.perl -l ${src} > ${data_tmp}/${file}.prepros.${src}
  cat ${data_tmp}/${file}.original.clean.${tgt} |
    ${moses_decoder}/tokenizer/normalize-punctuation.perl -l ${src} |
    ${moses_decoder}/tokenizer/tokenizer.perl -l ${src} > ${data_tmp}/${file}.prepros.${tgt}
done

## language calssification using fastext
#for file in ${FILES[@]} 
#  do
#    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_tmp}/${file}.original.clean.${src} > ${data_tmp}/${file}.${src}.lang_classification.fast_text
#    ${fast_text}/fasttext predict ${fast_text}/language_recognition.bin ${data_tmp}/${file}.original.clean.${tgt} > ${data_tmp}/${file}.${tgt}.lang_classification.fast_text
#
#    ### for marianNMT
#    paste ${data_tmp}/${file}.prepros.${src} ${data_tmp}/${file}.prepros.${tgt} ${data_tmp}/${file}.${src}.lang_classification.fast_text \
#    | awk -F'\t' '{if ($3=="__label__es") {print $1 > "'${data_tmp}'/'${file}'.filtered.'${src}'"; print $2 > "'${data_tmp}'/'${file}'.filtered.'${tgt}'"}}'
#
#    ${moses_decoder}/training/clean-corpus-n.perl ${data_tmp}/${file}.prepros ${src} ${tgt} ${data_tmp}/${file}.prepros.clean 1 120
#    ${moses_decoder}/training/clean-corpus-n.perl ${data_tmp}/${file}.filtered ${src} ${tgt} ${data_tmp}/${file}.filtered.clean 1 120
#done


## add the ratio
for file in ${FILES[@]} 
  do
  #  python3 ${python_scripts}/preprocess/lev_ratio.py ${data_tmp}/${file}.filtered.clean.${src} ${data_tmp}/${file}.filtered.clean.${tgt} ${data_tmp}/${file}.filtered.lev.${src} ${data_tmp}/${file}.filtered.lev.${tgt}
    python3 ${python_scripts}/preprocess/lev_ratio.py ${data_tmp}/${file}.original.clean.${src} ${data_tmp}/${file}.original.clean.${tgt} ${data_tmp}/${file}.original.lev.${src} ${data_tmp}/${file}.original.lev.${tgt}
done

for file in ${FILES[@]}
do
  cat ${data_tmp}/${file}.original.lev.${src} >> ${data_output}/train.original.lev.${src}
  cat ${data_tmp}/${file}.original.lev.${tgt} >> ${data_output}/train.original.lev.${tgt}
  ## 
 # cat ${data_tmp}/${file}.prepros.clean.${src} >> ${data_output}/train.prepros.${src}
 # cat ${data_tmp}/${file}.prepros.clean.${tgt} >> ${data_output}/train.prepros.${tgt}
  ###
 # cat ${data_tmp}/${file}.filtered.clean.${src} >> ${data_output}/train.filtered.${src}
 # cat ${data_tmp}/${file}.filtered.clean.${tgt} >> ${data_output}/train.filtered.${tgt}
  ###   
 # cat ${data_tmp}/${file}.filtered.lev.${src} >> ${data_output}/train.filtered.lev.${src}
 # cat ${data_tmp}/${file}.filtered.lev.${tgt} >> ${data_output}/train.filtered.lev.${tgt}
done

rm -r ${data_tmp}

