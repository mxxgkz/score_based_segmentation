#!/bin/bash
while IFS=$' ' read IDX IMGLEN WINDH WINDW EWMASIG EWMAWIND ZSCALE NOISSIG NOISSIZE NCOMP TRRD IMG_F IMG_TR IMG_VAL IMG_PI IMG_PII PII_POR LSSIM LSBLK_S LSBLK_IDX
do
STD_OUTPUT_FILE="../../std_output/${IDX}_real_reg_2d_non_causal_2.txt"
RES_FOLDER_PATH="./Experiments/Octyl_examples/figures/"
# RES_FOLDER_PATH="./Experiments/Dual_phase_examples/figures/"

JOB=`sbatch << EOJ
#!/bin/bash
#SBATCH -J ${IDX}
#SBATCH -A p30309
#SBATCH -p normal
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=110G #MB # Total memory
#SBATCH --output=${STD_OUTPUT_FILE}
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=zkghhg@gmail.com

#Delete any preceding space after 'EOJ'. OW, there will be some error.

# # add a project directory to your PATH (if needed)
# export PATH=$PATH:/projects/p20XXX/tools/

# unload any modules that carried over from your command line session
module purge

# Set your working directory
cd /projects/p30309/neurips2021/Nonstationarity_Diagnostics/   #$PBS_O_WORKDIR

# load modules you need to use
module load python/anaconda3.6
module load ghostscript

source activate py37

# # A command you actually want to execute:
# java -jar <someinput> <someoutput>
# # Another command you actually want to execute, if needed:
# python myscript.py

# # For autoregressive 1d linear regression.
# python -uB single_sim_call.py --model_file_folder="${RES_FOLDER_PATH}sim_0_controlled_ala_rate_0_99/" --nnet=0 --alarm_level=99 --single_exp_plot=21 --model_idx=4 --reg_model='nnet_lin' --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --eff_wind_len_factor=2 --PII_len=20000

# # For autoregressive 2d linear regression. 
# python -uB single_sim_call.py --model_file_folder="${RES_FOLDER_PATH}sim_1_${IDX}_ewmasig_${EWMASIG}_ewmawind_${EWMAWIND}/" --nnet=0 --alarm_level=99.9 --single_exp_plot=23 --model_idx=5 --reg_model='nnet_lin' --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation=sigmoid --eff_wind_len_factor=2 --PII_len=20000 --img_hei=200 --img_wid=200 --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND}

# For autoregressive 2d neural-net regression. Cross-validation.
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_cv_non_causal_retro/' --nnet=1 --alarm_level=99.0 --single_exp_plot=31 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=1 --cv_n_jobs=30 --cv_N_rep=3 --cv_K_fold=5 --cv_rand_search=60 --cv_pre_dispatch='1.2*n_jobs' --cv_task_param_ls='penal_param-training_batch_size-learning_rate' --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=1 --n_comp=${NCOMP} --real_img_folder='/home/ghhgkz/scratch/Data/texture/Prof_Brinson/data1/'

# # For autoregressive 2d neural-net regression. No cross-validation. Prospective.
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_no_cv_non_causal/' --nnet=0 --alarm_level=99.0 --single_exp_plot=31 --model_idx=5 --reg_model='lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=1 --n_comp=${NCOMP} --real_img_folder=${IMG_F} --train_img=${IMG_TR} --val_img=${IMG_VAL} --PI_img=${IMG_PI} --PII_img=${IMG_PII}

# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_no_cv_non_causal/' --nnet=1 --alarm_level=99.0 --single_exp_plot=31 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=3 --n_comp=${NCOMP} --real_img_folder=${IMG_F} --train_img=${IMG_TR} --val_img=${IMG_VAL} --PI_img=${IMG_PI} --PII_img=${IMG_PII}

# # For autoregressive 2d neural-net regression. No cross-validation. Prospective. Separate analysis for images in Phase-II so no ariticial boundary effects.
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_no_cv_non_causal/' --nnet=0 --alarm_level=99.0 --single_exp_plot=313131 --model_idx=5 --reg_model='lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=1 --n_comp=${NCOMP} --real_img_folder=${IMG_F} --train_img=${IMG_TR} --val_img=${IMG_VAL} --PI_img=${IMG_PI} --PII_img=${IMG_PII} --PII_img_ls=${LS_IMG_PII}

# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_no_cv_non_causal/' --nnet=1 --alarm_level=99.0 --single_exp_plot=313131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=3 --n_comp=${NCOMP} --real_img_folder=${IMG_F} --train_img=${IMG_TR} --val_img=${IMG_VAL} --PI_img=${IMG_PI} --PII_img=${IMG_PII} --PII_img_ls=${LS_IMG_PII}

# # Integrate the reading image into the simulation script. We read data from files, instead of simulating images. can be used for replot.
# Linear model
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_no_cv_non_causal/' --nnet=0 --alarm_level=99.0 --single_exp_plot=31313131 --model_idx=5 --reg_model='lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=2 --gen_wind_wid=2 --materials_model="non_causal" --training_rounds=${TRRD} --n_comp=${NCOMP} --real_img_folder=${IMG_F} --train_img=${IMG_TR} --val_img=${IMG_VAL} --PI_img=${IMG_PI} --PII_img=${IMG_PII} --PII_img_ls=${LS_IMG_PII} --gen_img_data=0 --plot_img_PI_idx_str=${PI_IDX} --plot_img_PII_idx_str=${PII_IDX} --plot_metric_idx_str=${MET_IDX}

# # Nnet model
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_no_cv_non_causal/' --nnet=1 --alarm_level=99.0 --single_exp_plot=31313131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=2 --gen_wind_wid=2 --materials_model="non_causal" --training_rounds=${TRRD} --n_comp=${NCOMP} --real_img_folder=${IMG_F} --train_img=${IMG_TR} --val_img=${IMG_VAL} --PI_img=${IMG_PI} --PII_img=${IMG_PII} --PII_img_ls=${LS_IMG_PII} --gen_img_data=0 --plot_img_PI_idx_str=${PI_IDX} --plot_img_PII_idx_str=${PII_IDX} --plot_metric_idx_str=${MET_IDX}

# Cross-validation
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_cv_non_causal/' --nnet=1 --alarm_level=99.0 --single_exp_plot=313131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=1 --cv_n_jobs=60 --cv_task_param_ls="penal_param-training_batch_size-learning_rate" --cv_N_rep=3 --cv_K_fold=5 --cv_rand_search=60 --cv_pre_dispatch="1*n_jobs" --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=3 --n_comp=${NCOMP} --real_img_folder=${IMG_F} --train_img=${IMG_TR} --val_img=${IMG_VAL} --PI_img=${IMG_PI} --PII_img=${IMG_PII} --PII_img_ls=${LS_IMG_PII}

# # For autoregressive 2d neural-net regression. No cross-validation. Prospective multiple images. Separate analysis for images in Phase-II so no ariticial boundary effects. For replot purpose.
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_no_cv_non_causal/' --nnet=0 --alarm_level=99.0 --single_exp_plot=31313131 --model_idx=5 --reg_model='lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=2 --gen_wind_wid=2 --materials_model="non_causal" --training_rounds=1 --n_comp=${NCOMP} --ar_model_coef_folder_path=${IMG_F} --train_PI_coeff=${IMG_TR} --PII_coeff=${IMG_PII} --num_train_imgs=4 --num_val_imgs=4 --num_PI_imgs=4 --num_PII_imgs=4 

# # --rand_seed=5994 #23545

# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_no_cv_non_causal/' --nnet=1 --alarm_level=99.0 --single_exp_plot=31313131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=2 --gen_wind_wid=2 --materials_model="non_causal" --training_rounds=3 --n_comp=${NCOMP} --ar_model_coef_folder_path=${IMG_F} --train_PI_coeff=${IMG_TR} --PII_coeff=${IMG_PII} --num_train_imgs=4 --num_val_imgs=4 --num_PI_imgs=4 --num_PII_imgs=4

# # Run the single simulation
# # Linear model
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_no_cv_non_causal/' --nnet=0 --alarm_level=99.0 --single_exp_plot=31313131 --model_idx=5 --reg_model='lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=2 --gen_wind_wid=2 --materials_model="non_causal" --training_rounds=1 --n_comp=${NCOMP} --ar_model_coef_folder_path=${IMG_F} --train_PI_coeff=${IMG_TR} --PII_coeff=${IMG_PII} --num_train_imgs=1 --num_val_imgs=1 --num_PI_imgs=12 --num_PII_imgs=4 --PII_portion=${PII_POR} --noise_level_lambda=-1

# # Nnet model
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_no_cv_non_causal/' --nnet=1 --alarm_level=99.0 --single_exp_plot=31313131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=2 --gen_wind_wid=2 --materials_model="non_causal" --training_rounds=3 --n_comp=${NCOMP} --ar_model_coef_folder_path=${IMG_F} --train_PI_coeff=${IMG_TR} --PII_coeff=${IMG_PII} --num_train_imgs=1 --num_val_imgs=1 --num_PI_imgs=12 --num_PII_imgs=4 --PII_portion=${PII_POR} --noise_level_lambda=-1

# For autoregressive or real image 2d neural-net regression. Cross-validation. Retrospective analysis.

python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_cv_non_causal_retro/' --nnet=1 --alarm_level=99.0 --single_exp_plot=3131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=1 --cv_n_jobs=30 --cv_N_rep=3 --cv_K_fold=5 --cv_rand_search=60 --cv_pre_dispatch='1.2*n_jobs' --cv_task_param_ls='penal_param-training_batch_size-learning_rate' --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=${TRRD} --n_comp=${NCOMP} --real_img_path='${IMG_PII}'

# For autoregressive or real image 2d neural-net regression. No cross-validation. Retrospective analysis.

# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_no_cv_non_causal_retro/' --nnet=0 --alarm_level=99.0 --single_exp_plot=3131 --model_idx=5 --reg_model='lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=1 --n_comp=${NCOMP} --real_img_path='${IMG_PII}'

# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet_no_cv_non_causal_retro/' --nnet=1 --alarm_level=99.0 --single_exp_plot=3131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --cv_flag=0 --max_steps=10000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1 --training_rounds=${TRRD} --n_comp=${NCOMP} --penal_param=0.01 --learning_rate=1e-05 --stopping_lag=1000 --training_batch_size=100 --real_img_path='${IMG_PII}'

# For autoregressive 2d neural-net classification. With cross-validation.
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim2_cla_nnet_intcp_0_cv_causal_zscale_retro_img_len_${IMGLEN}/' --nnet=1 --alarm_level=99.0 --single_exp_plot=24 --model_idx=5 --reg_model='nnet_logi' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --clf_thr=0.47416 --cv_flag=1 --cv_n_jobs=30 --cv_task_param_ls="penal_param-training_batch_size-learning_rate" --cv_rand_search=60 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="causal" --nois_profile_sigma=2 --nois_size=100 --nois_scale=1

# For autoregressive 2d neural-net classification. No cross-validation.
# Latent variable as X: (0, 0.4991) (0.5, 0.60) (1, 069314) (2, 0.69314)
# Obersvation as X: (0, 0.47416) (0.1, 0.55338) (0.5, 0.58535) (1, 0.69082) (2, 0.85594)

# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim2_cla_intcp_0_no_cv_non_causal_zscale_retro_img_len_${IMGLEN}/' --nnet=0 --alarm_level=99.0 --single_exp_plot=2424 --model_idx=5 --reg_model='logi' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --clf_thr=-1 --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1

# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim2_cla_nnet_intcp_0_no_cv_non_causal_zscale_retro_img_len_${IMGLEN}/' --nnet=1 --alarm_level=99.0 --single_exp_plot=2424 --model_idx=5 --reg_model='nnet_logi' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --clf_thr=0.47416 --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="non_causal" --nois_profile_sigma=${NOISSIG} --nois_size=${NOISSIZE} --nois_scale=1


# To purposely overfit neural network to reach as high training accuracy as possible, to reconstruct stochastic materials structure.
# Latent variable as X: (0, 0.4991) (0.5, 0.60) (1, 069314) (2, 0.69314)
# Obersvation as X: (0, 0.47416) (0.1, 0.55338) (0.5, 0.58535) (1, 0.69082) (2, 0.85594)
# python -uB single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}regen_overfit_${IDX}_sim2_cla_nnet_intcp_0_no_cv_causal_1_zscale_img_len_${IMGLEN}/' --nnet=1 --alarm_level=99.0 --single_exp_plot=24 --model_idx=5 --reg_model='nnet_logi' --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --nois_sigma=0.1 --intcp=0 --z_scale=${ZSCALE} --clf_thr=0.47416 --cv_flag=0 --max_steps=50000 --stopping_lag=2000 --activation='sigmoid' --img_hei=${IMGLEN} --img_wid=${IMGLEN} --gen_wind_hei=4 --gen_wind_wid=4 --materials_model="causal_1" --learning_rate=0.01 --penal_param=0 --training_rounds=5 --decay_steps=1000

#########################################################
# Previous experiments of retrospective analysis.

# For old experiments.
# python single_sim_call.py --model_file_folder="../Exper_Res/20190918-Try_CD_Materials_Detection/figures/train_causal_${WINDH}_${WINDW}_${EWMASIG}_${EWMAWIND}_cv_1/" --materials_model="causal" --nnet=1 --single_exp_plot=-2 --reg_model=nnet_lin --cv_flag=1 --cv_n_jobs=60 --cv_N_rep=3 --cv_K_fold=5 --cv_rand_search=60 --cv_pre_dispatch="0.15*n_jobs" --max_steps=40000 --stopping_lag=1000 --activation="sigmoid" --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --data_file_folder="../Exper_Res/20190918-Try_CD_Materials_Detection/figures/" --PI_img_file_name="ref_mat_sample_bw_50.tif" --PII_img_file_name="targ_mat_sample_bw_50.tif"

# For cross-validation.
# python single_sim_call.py --model_file_folder="../Exper_Res/20190928-Gen_GP_Rand_Struct/figures/train_causal_${WINDH}_${WINDW}_${EWMASIG}_${EWMAWIND}_cv_1/" --materials_model="causal" --nnet=1 --single_exp_plot=-2 --reg_model=nnet_lin --cv_flag=1 --cv_n_jobs=60 --cv_N_rep=3 --cv_K_fold=5 --cv_rand_search=60 --cv_pre_dispatch="0.2*n_jobs" --max_steps=40000 --stopping_lag=1000 --activation="sigmoid" --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --data_file_folder="../Exper_Res/20190928-Gen_GP_Rand_Struct/figures/" --PI_img_file_name="ref_img_binary_gp.png" --PII_img_file_name="targ_img_binary_gp.png" --regen_grid_size=200


# For experiments to find out window size and exponential weight scales.
# python single_sim_call.py --model_file_folder="../Exper_Res/20190928-Gen_GP_Rand_Struct/figures/train_causal_${WINDH}_${WINDW}_${EWMASIG}_${EWMAWIND}/" --materials_model="causal" --nnet=1 --single_exp_plot=-2 --reg_model=nnet_lin --cv_flag=0 --max_steps=40000 --stopping_lag=1000 --activation="sigmoid" --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --data_file_folder="../Exper_Res/20190928-Gen_GP_Rand_Struct/figures/" --PI_img_file_name="ref_img_binary_gp.png" --PII_img_file_name="targ_img_binary_gp.png" --regen_grid_size=200


# For classification model for binary images for micro-structure. Cross-validation.
# python single_sim_call.py --model_file_folder="../Exper_Res/20190929-Gen_GP_Rand_Struct_Cla/figures/train_causal_${WINDH}_${WINDW}_${EWMASIG}_${EWMAWIND}_cv_1/" --materials_model="causal" --nnet=1 --single_exp_plot=-3 --reg_model=nnet_lin --cv_flag=1 --cv_n_jobs=60 --cv_N_rep=3 --cv_K_fold=5 --cv_rand_search=60 --cv_pre_dispatch="0.2*n_jobs" --max_steps=40000 --stopping_lag=1000 --activation="sigmoid" --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --data_file_folder="../Exper_Res/20190929-Gen_GP_Rand_Struct_Cla/figures/" --PI_img_file_name="ref_img_binary_gp.png" --PII_img_file_name="targ_img_binary_gp.png" --regen_grid_size=200

# For classification model for binary images for micro-structure.
# python single_sim_call.py --model_file_folder="../Exper_Res/20190929-Gen_GP_Rand_Struct_Cla/figures/train_causal_${WINDH}_${WINDW}_${EWMASIG}_${EWMAWIND}/" --materials_model="causal" --nnet=1 --single_exp_plot=-3 --reg_model=nnet_lin --cv_flag=0 --max_steps=40000 --stopping_lag=1000 --activation="sigmoid" --wind_hei=${WINDH} --wind_wid=${WINDW} --spatial_ewma_sigma=${EWMASIG} --spatial_ewma_wind_len=${EWMAWIND} --data_file_folder="../Exper_Res/20190929-Gen_GP_Rand_Struct_Cla/figures/" --PI_img_file_name="ref_img_binary_gp.png" --PII_img_file_name="targ_img_binary_gp.png" --regen_grid_size=200

EOJ
`

# print out the job id for reference later
echo "JobID = ${JOB} for indices ${IDX} and parameters ${WINDH}, ${WINDW}, ${EWMASIG}, ${EWMAWIND}, ${ZSCALE} submitted on `date`"

sleep 1

done < ./command_script/param_real_reg.info
exit

# make this file executable and then run from the command line
# chmod u+x submit.sh
# ./submit.sh
# The last line of params.txt have to be an empty line.