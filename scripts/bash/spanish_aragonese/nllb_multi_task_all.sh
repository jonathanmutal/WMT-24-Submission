data_path=./experiments/spanish_aragonese/data

outputdir="./output/nllb/spanish_aragonese/multi_task"
dev_path="./data/FLORES+/dev_task/"
json_with_data="{\"fra_Latn\": [{\"source\": \"./data/monolingual/aragonese/clean.arg-es.es\", \"target\": \"./data/monolingual/aragonese/clean.arg-es.arg\"}, {\"source\": \"${data_path}/train.filter.optimal.es\", \"target\": \"${data_path}/train.filter.optimal.an\"}], \"ita_Latn\": [{\"source\": \"./data/monolingual/aragonese/clean.arg-es.artificial.arg\", \"target\": \"./data/monolingual/aragonese/clean.arg-es.arg\"}, {\"source\": \"${data_path}/train.filter.artificial.optimal.an\", \"target\": \"${data_path}/train.filter.optimal.an\"}, {\"source\": \"${data_path}/pilar.arg-es.artificial.arg\", \"target\": \"${data_path}/pilar.arg-es.arg\"}, {\"source\": \"./data/monolingual/spanish/clean.es-arg.es\", \"target\": \"./data/monolingual/spanish/clean.es-arg.arg\"}]}"
json_validation="{\"ita_Latn\":{\"source\": \"${dev_path}/dev.artificial.arg_Latn\", \"target\": \"${dev_path}/dev.arg_Latn\"}}"

lrs=(9e-8 1e-7)
if [ ! -d ${outputdir} ]
then
  mkdir -p ${outputdir}
fi

#data=all.artificial
data=pilar.arg-es
seed=111
tgt_lang=fra_Latn

from_model=facebook/nllb-200-distilled-600M


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
     sbatch ./scripts/slrum/python/train/nllb_multitask_resume.sh ${outputdir}/${lr} ${data_path}/${data}.es ${data_path}/${data}.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" "${json_validation}" ${outputdir}/${lr} ${outputdir}/${lr}
  else
    echo "training ${outputdir}/${lr}"
    sbatch ./scripts/slrum/python/train/nllb_multitask.sh ${outputdir}/${lr} ${data_path}/${data}.es ${data_path}/${data}.arg ${dev_path}/dev.spa_Latn ${dev_path}/dev.arg_Latn ${lr} ${seed} spa_Latn ${tgt_lang} "${json_with_data}" "${json_validation}" ${from_model}
    fi
done
