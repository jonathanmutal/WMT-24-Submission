data_path=./experiments/spanish_aragonese/data
data=./data/FLORES+/dev_task/

outputdir="./output/nllb/spanish_aragonese/from_aragonese_baseline/final_models_best"
dev_path="./data/FLORES+/dev_task/"

from_model="./output/nllb/spanish_aragonese/aragonese_baseline_100tok/1e-5/bestmodel_170000"

if [ ! -d ${outputdir} ]
then
  mkdir -p ${outputdir}
fi

seed=111
tgt_lang=fra_Latn

lrs=(9e-6)
for lr in ${lrs[@]}
do
  if [ ! -d ${outputdir}/${lr} ]
  then
    echo "creating ${outputdir}/${lr}"
    mkdir -p ${outputdir}/${lr}
  fi
  job_id=$(sbatch -J ${lr}dev ./scripts/slrum/python/train/nllb_small_steps.sh ${outputdir}/${lr} ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} ${from_model})  
#   sbatch -J ${lr}dev ./scripts/slrum/python/train/nllb_resume_small_steps.sh ${outputdir}/${lr} ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} ${outputdir}/${lr} ${outputdir}/${lr}
done
