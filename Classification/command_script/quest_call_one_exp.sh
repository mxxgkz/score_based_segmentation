#!/bin/bash
while IFS=$' ' read IDX CID MOD PWEI NORMP BBONE WEINAME CORE HOUR EPOCH POSTFIX DBFN NEWFN GENFD START START_P SWEI NORMD WDEC
do
STD_OUTPUT_FILE="../../std_output/${IDX}_quest_${POSTFIX}.log"
RES_FOLDER_PATH="./Experiments/Kylberg_examples/figures/"


JOB=`sbatch << EOJ
#!/bin/bash
#SBATCH -J ${IDX}
#SBATCH -A p30309
#SBATCH -p normal
#SBATCH -t ${HOUR}:59:59
#SBATCH -N 1
#SBATCH -n ${CORE}
#SBATCH --mem=55768 #MB
#SBATCH --output=${STD_OUTPUT_FILE}
#SBATCH --mail-type=FAIL,BEGIN,END #BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=zkghhg@gmail.com

#Delete any preceding space after 'EOJ'. OW, there will be some error.

# unload any modules that carried over from your command line session
module purge

# Set your working directory
cd /projects/p30309/neurips2021/Classification/   #$PBS_O_WORKDIR

# load modules you need to use
module load python/anaconda3.6
source activate py37

# # A command you actually want to execute:
# java -jar <someinput> <someoutput>
# # Another command you actually want to execute, if needed:
# python myscript.py

python -uB ./finetune.py --model_name=${MOD} --weights_path=${WEINAME} --num_epochs=${EPOCH} --postfix=${POSTFIX} --n_threads=${CORE} --norm_data=${NORMD} --weight_decay=${WDEC}

EOJ
`

# print out the job id for reference later
echo "JobID = ${JOB} for indices ${IDX} and parameters ${EPOCH}, ${POSTFIX}, ${START}, ${START_P} submitted on `date`"

sleep 0.5

done < ./command_script/param_classification_exp.info
exit

# make this file executable and then run from the command line
# chmod u+x submit.sh
# ./submit.sh
# The last line of params.txt have to be an empty line.
