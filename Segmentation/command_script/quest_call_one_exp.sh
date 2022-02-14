#!/bin/bash
while IFS=$' ' read IDX CID MOD PWEI NORMP BBONE WEINAME CORE HOUR LIDXVAL POSTFIX DBFN NEWFN GENFD START START_P SWEI NORMD WDEC SFIDX EFIDX ACSTEP INITVAL_STEP RANDS
do
STD_OUTPUT_FILE="../../std_output/${IDX}_expanse_${SFIDX}_${ACSTEP}.log"


JOB=`sbatch << EOJ
#!/bin/bash
#SBATCH -J ${IDX}
#SBATCH -A p30309
#SBATCH -p normal
#SBATCH -t ${HOUR}:59:59
#SBATCH -N 1
#SBATCH -n ${CORE} # cpu
#SBATCH --mem=110G
#SBATCH --output=${STD_OUTPUT_FILE}
#SBATCH --mail-type=BEGIN,FAIL,END,REQUEUE #BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=zkghhg@gmail.com

#Delete any preceding space after 'EOJ'. OW, there will be some error.

# unload any modules that carried over from your command line session
module purge

# Set your working directory
cd /projects/p30309/neurips2021/Segmentation/   #$PBS_O_WORKDIR

# load modules you need to use
module load python/anaconda3.6
source activate py37

# set slurm
# python -uB ./copy_file_local.py

# # A command you actually want to execute:
# java -jar <someinput> <someoutput>
# # Another command you actually want to execute, if needed:
# python myscript.py

# python -uB ./finetune_rand_init_full_training.py

# python -uB ./finetune_rand_init_full_training_incremental_stepwise.py --num_epochs=${LIDXVAL} --postfix=${POSTFIX} --start_rd=${START} --start_model_path=${START_P} --db_fd_name=${DBFN} --new_fd_name=${NEWFN}

# python -uB ./finetune_rand_init_full_training_incremental_stepwise.py --model_name=${MOD} --weights_name=${WEINAME} --num_epochs=${LIDXVAL} --postfix=${POSTFIX} --start_rd=${START} --start_model_path=${START_P} --db_fd_name=${DBFN} --new_fd_name=${NEWFN} --gen_fd_prefix=${IDX} --n_threads=${CORE} --backbone=${BBONE} --pwei_flag=${PWEI} --normp=${NORMP}

python -uB ./finetune_homo_texture.py --model_name=${MOD} --weights_name=${WEINAME} --num_epochs=${LIDXVAL} --postfix=${POSTFIX} --start_rd=${START} --start_model_path=${START_P} --db_fd_name=${DBFN} --new_fd_name=${NEWFN} --gen_fd_prefix=${IDX} --n_threads=${CORE} --backbone=${BBONE} --pwei_flag=${PWEI} --normp=${NORMP}

# python -uB ./finetune_dendrites.py --model_name=${MOD} --num_epochs=${LIDXVAL} --postfix=${POSTFIX} --start_rd=${START} --start_model_path=${START_P} --gen_fd_prefix=${IDX} --n_threads=${CORE} --backbone=${BBONE} --dendrites_data=${GENFD}

# python -uB ./finetune_cv_deeplab3plus_seg.py --model_name=${MOD} --last_idx_valids=${LIDXVAL} --postfix=${POSTFIX} --start_rd=${START} --start_model_path=${START_P} --gen_fd_prefix=${IDX} --n_threads=${CORE} --backbone=${BBONE} --sample_wei_flag=${SWEI} --norm_data=${NORMD} --weight_decay=${WDEC} --modified_pre_trained_path=${WEINAME} --start_fold_idx=${SFIDX} --end_fold_idx=${EFIDX} --accu_step=${ACSTEP} --init_valid_step=${INITVAL_STEP} --rand_seed=${RANDS}

# python -uB ./load_partial_weights.py

EOJ
`

# print out the job id for reference later
echo "JobID = ${JOB} for indices ${IDX} and parameters ${LIDXVAL}, ${POSTFIX}, ${START}, ${START_P} submitted on `date`"

sleep 0.5

done < ./command_script/param_segmentation_exp.info
# done < ./command_script/param_unet_exp.info
# done < ./command_script/param_trex_online.info
# done < ./command_script/param_ar_lin_2d.info
exit

# make this file executable and then run from the command line
# chmod u+x submit.sh
# ./submit.sh
# The last line of params.txt have to be an empty line.
