outputdir="./output/nllb/spanish_occitan/best_config/final_models/from_multilingual_dev"

data_path=./experiments/spanish_occitan/data

if [ ! -d ${outputdir} ]
then
  mkdir -p ${outputdir}
fi

data=all.artificial
seed=111
tgt_lang=oci_Latn
dev_path="./data/FLORES+/dev_task"
from_model="./output/nllb/spanish_occitan/best_config/all_multilingual_3_100tok_synth/1e-5/bestmodel_110000"


#lrs=(1e-8 5e-8 9e-6)
#lrs=(1e-6 3e-6)
lrs=(9e-6)
for lr in ${lrs[@]}
do
  if [ ! -d ${outputdir}/${lr} ]
  then
    echo "creating ${outputdir}/${lr}"
    mkdir -p ${outputdir}/${lr}
  fi
  echo "skiping"
#  sbatch -J ${lr}dev ./scripts/slrum/python/train/nllb_small_steps.sh ${outputdir}/${lr} ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} ${from_model}
  sbatch -J ${lr}ARNdev ./scripts/slrum/python/train/nllb_resume_small_steps.sh ${outputdir}/${lr} ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${dev_path}/dev.spa_Latn ${dev_path}/dev.arn_Latn ${lr} ${seed} spa_Latn ${tgt_lang} ${outputdir}/${lr} ${outputdir}/${lr}
done
