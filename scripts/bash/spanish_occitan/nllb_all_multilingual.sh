outputdir="./output/nllb/spanish_occitan/best_config/all_multilingual_3_100tok_synth"

data_path=./experiments/spanish_occitan/data

if [ ! -d ${outputdir} ]
then
  mkdir -p ${outputdir}
fi

#data=all.artificial
# dataset 2
data=all.artificial
seed=111
tgt_lang=oci_Latn
# dataset cat_Latn: -1 -3 -4 5 -6
json_with_data="{\"cat_Latn\": [{\"source\": \"${data_path}/train.original.es\", \"target\": \"${data_path}/train.original.oc\"}, {\"source\": \"./data/monolingual/occitan/clean.oc-es.es\", \"target\": \"./data/monolingual/occitan/clean.oc-es.oc\"}, {\"source\": \"./data/monolingual/spanish/clean.es-oc.es\", \"target\": \"./data/monolingual/spanish/clean.es-oc.oc\"}, {\"source\": \"./data/monolingual/spanish/wikimedia.es-oc.es\", \"target\": \"./data/monolingual/spanish/wikimedia.es-oc.oc\"}, {\"source\": \"${data_path}/train.artificial.target.es\", \"target\": \"${data_path}/train.artificial.target.oc\"}, {\"source\": \"./data/monolingual/spanish/clean.es-an.es\", \"target\": \"./data/monolingual/spanish/clean.es-an.an\"}, {\"source\": \"./data/monolingual/spanish/wikimedia.es-an.es\", \"target\": \"./data/monolingual/spanish/wikimedia.es-an.an\"}], \"oci_Latn\": [{\"source\": \"./data/monolingual/occitan/clean.synthetic.bloomz.60k.random.es\",\"target\": \"./data/monolingual/occitan/clean.synthetic.bloomz.60k.random.arn\"}]}"
dev_path="./data/FLORES+/dev_task"
from_model=facebook/nllb-200-distilled-600M


#lrs=(3e-6 1e-5 3e-5 9e-6)
#lrs=(1e-6 3e-6 9e-6 1e-5)
lrs=(9e-6)
for lr in ${lrs[@]}
do
  if [ ! -d ${outputdir}/${lr} ]
  then
    echo "creating ${outputdir}/${lr}"
    mkdir -p ${outputdir}/${lr}
  fi
  if compgen -G ${outputdir}/${lr}/{step,bestmodel}_*
  then
     echo "skiping"
     job_id=$(sbatch -J ${lr}msyn ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
  else
    echo "training ${outputdir}/${lr}/${data}"
    job_id=$(sbatch -J ${lr}msyn ./scripts/slrum/python/train/nllb_multilingual.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${from_model})
  fi
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}msyn --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}msyn --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}msyn --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr}) 
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}msyn --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr}) 
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}msyn --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr}) 
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}msyn --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr}) 
done
