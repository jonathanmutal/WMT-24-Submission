data_path=./data/monolingual/spanish/

outputdir="./output/nllb/spanish_aragonese/aragonese_baseline_100tok_all"
dev_path="./data/FLORES+/dev_task/"

json_with_data="{\"fra_Latn\": [{\"source\": \"${data_path}/clean_1.es-arg.es\", \"target\": \"${data_path}/clean_1.es-arg.arg\"}, {\"source\": \"./data/monolingual/aragonese/clean.arg-es.es\", \"target\": \"./data/monolingual/aragonese/clean.arg-es.arg\"}, {\"source\": \"./experiments/spanish_aragonese/data/train.original.es\", \"target\": \"./experiments/spanish_aragonese/data/train.original.an\"}, {\"source\": \"./experiments/spanish_aragonese/data/pilar.arg-es.es\", \"target\": \"./experiments/spanish_aragonese/data/pilar.arg-es.arg\"}, {\"source\": \"./data/monolingual/aragonese/synthetic_bloomz.clean.arg-es.es\", \"target\": \"./data/monolingual/aragonese/synthetic_bloomz.clean.arg-es.arg\"}]}"

#json_with_data="{\"fra_Latn\": []}"

if [ ! -d ${outputdir} ]
then
  mkdir -p ${outputdir}
fi

seed=111
tgt_lang=fra_Latn

from_model=facebook/nllb-200-distilled-600M


lrs=(1e-5 9e-6 3e-6 9e-5)
#lrs=(9e-5)
for lr in ${lrs[@]}
do
  if [ ! -d ${outputdir}/${lr} ]
  then
    echo "creating ${outputdir}/${lr}"
    mkdir -p ${outputdir}/${lr}
  fi
  if compgen -G ${outputdir}/${lr}/*{step,bestmodel}_*
  then
     echo "skiping"
     job_id=$(sbatch -J araB${lr} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/wikimedia.es-arg.es ${data_path}/wikimedia.es-arg.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
  else
    echo "training ${outputdir}/${lr}"
    job_id=$(sbatch -J araB${lr} ./scripts/slrum/python/train/nllb_multilingual.sh ${outputdir}/${lr} ${data_path}/wikimedia.es-arg.es ${data_path}/wikimedia.es-arg.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${from_model})
  fi
    job_id=$(echo ${job_id} | awk -F" " '{print $4}')
    job_id=$(sbatch -J araB${lr} --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/wikimedia.es-arg.es ${data_path}/wikimedia.es-arg.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
    job_id=$(echo ${job_id} | awk -F" " '{print $4}')
    job_id=$(sbatch -J araB${lr} --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/wikimedia.es-arg.es ${data_path}/wikimedia.es-arg.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
    job_id=$(echo ${job_id} | awk -F" " '{print $4}')
    job_id=$(sbatch -J araB${lr} --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/wikimedia.es-arg.es ${data_path}/wikimedia.es-arg.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
    job_id=$(echo ${job_id} | awk -F" " '{print $4}')
    job_id=$(sbatch -J araB${lr} --dependency=afterany:${job_id} ./scripts/slrum/python/train/nllb_multilingual_resume.sh ${outputdir}/${lr} ${data_path}/wikimedia.es-arg.es ${data_path}/wikimedia.es-arg.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" ${outputdir}/${lr} ${outputdir}/${lr})
done
