outputdir="./output/nllb/multilingual.original.artificial.target"

data_path=./experiments/spanish_occitan/data

if [ ! -d ${outputdir} ]
then
  mkdir -p ${outputdir}
fi

data=all.artificial
seed=111
tgt_lang=oci_Latn
lrs=(2e-5 4e-5 5e-5 6e-5 1e-4)
for lr in ${lrs[@]}
do
  if [ ! -d ${outputdir}/${lr}/${data}/${tgt_lang} ]
  then
    echo "creating ${outputdir}/${lr}/${data}/${tgt_lang}"
    mkdir -p ${outputdir}/${lr}/${data}/${tgt_lang}
  fi
  if compgen -G ${outputdir}/${lr}/${data}/${tgt_lang}/{step_,bestmodel_}*
  then
     echo "continue" 
     path_to_model=$(ls -td ${outputdir}/${lr}/${data}/${tgt_lang}/{step_,bestmodel_}* | head -1)
     sbatch ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr}/${data}/${tgt_lang} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${data_path}/dev.1.es ${data_path}/dev.1.oc ${lr} ${seed} spa_Latn ${tgt_lang} "{\"cat_Latn\": [{\"source\": \"${data_path}/train.original.es\", \"target\": \"${data_path}/train.original.oc\"}, {\"source\": \"${data_path}/train.artificial.target.es\", \"target\": \"${data_path}/train.artificial.target.oc\"}]}" ${path_to_model} ${path_to_model}
  else
     echo "no file"
     sbatch ./scripts/slrum/python/train/nllb_multilingual.sh ${outputdir}/${lr}/${data}/${tgt_lang} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${data_path}/dev.1.es ${data_path}/dev.1.oc ${lr} ${seed} spa_Latn ${tgt_lang} "{\"cat_Latn\": [{\"source\": \"${data_path}/train.original.es\", \"target\": \"${data_path}/train.original.oc\"}, {\"source\": \"${data_path}/train.artificial.target.es\", \"target\": \"${data_path}/train.artificial.target.oc\"}]}"
  fi
done
