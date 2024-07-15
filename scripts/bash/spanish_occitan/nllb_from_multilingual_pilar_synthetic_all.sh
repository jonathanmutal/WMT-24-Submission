outputdir="./output/nllb/spanish_occitan/best_config/from_multilingual/pilar_synthetic_all_good"

if [ ! -d ${outputdir} ]
then
  mkdir -p ${outputdir}
fi

data_path=./experiments/spanish_occitan/data
data=all.artificial
seed=111
tgt_lang=oci_Latn
dev_path="./data/FLORES+/dev_task"
from_model="./output/nllb/spanish_occitan/best_config/all_multilingual_3_100tok/1e-5/bestmodel_175000"
json_with_data="{\"oci_Latn\": [{\"source\": \"./data/monolingual/occitan/clean.synthetic.bloomz.60k.random.es\",\"target\": \"./data/monolingual/occitan/clean.synthetic.bloomz.60k.random.arn\"}]}"

lrs=(1e-8 5e-8 1e-6 9e-6)
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
     job_id=$(sbatch -J ${lr}spA ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
  else
    echo "training ${outputdir}/${lr}"
    job_id=$(sbatch -J ${lr}spA ./scripts/slrum/python/train/nllb_multilingual.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${from_model})
    fi
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}spA --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}spA --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
  job_id=$(echo ${job_id} | awk -F" " '{print $4}')
  job_id=$(sbatch -J ${lr}spA --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/train.${data}.es ${data_path}/train.${data}.oc ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
done
