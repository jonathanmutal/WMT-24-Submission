#!/bin/sh
#SBATCH --job-name WMT           # this is a parameter to help you sort your job when listing it
#SBATCH --error train_multi_logs_%j.error     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output train_WMT_logs_%j.out      # optional. By default the error and output files are merged
#SBATCH --mem 35GB
#SBATCH --time 12:00:00                  # maximum run time
#SBATCH --partition shared-gpu
#SBATCH --gres=gpu:1

outputdir=$1
data_src=$2
data_tgt=$3
valid_src=$4
valid_tgt=$5
lr=$6
seed=$7

src_lang=$8
tgt_lang=$9

multilingual_json=${10}


GPUS=0
srun python ./scripts/python/train/seq2seq.py --train_source_file ${data_src} \
	                    --train_target_file ${data_tgt} \
				       --num_beams 5 \
				       --validation_source_file ${valid_src} \
				       --validation_target_file ${valid_tgt} \
				       --source_lang ${src_lang} \
				       --target_lang ${tgt_lang} \
				       --model_name_or_path facebook/nllb-200-distilled-600M \
                        --learning_rate ${lr} \
                        --num_warmup_steps 50 \
				       --per_device_train_batch_size 16 \
				       --per_device_eval_batch_size 8 \
				       --output_dir ${outputdir} \
				       --seed ${seed} \
				       --checkpointing_steps 5000 \
				       --with_tracking \
				       --overwrite_cache \
				       --num_train_epochs 10 \
				       --early_stopping \
				       --early_stopping_step 10 \
				       --max_source_length 64 \
				       --max_target_length 64 \
				       --multilingual \
				       --multilingual_files "${multilingual_json}"

