# A Framework for Supervised and Unsupervised Segmentation and Classification of Materials Microstructure Images

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Terminlogoy:
### HR (Homogeneous Region):
A region where microstructure stochastic nature is homogeneous.
### MC (Microstructure Class): 
A specific class of stochastic nature of a microstructure within a specific HR.

## Step 1: 
In an unsupervised manner, diagnose multiphase nonstationary behavior in micrographs using a recently-developed score-based nonstationarity diagnostic (ND) method. Specifically, we fit a single supervised learning model to a set of training micrograph(s), apply the model to predict each pixel of the training micrograph(s) to obtain score vectors pixel-by-pixel, and then cluster score vectors to segment HRs corresponding to distinct MCs.

To train the model(s) and plot results:

| Data set name         | Commend  |
| :------------------ | :---------------- |
| Silica particles in PMMA with octyl functional modification   |     python -uB ./Nonstationarity_Diagnostics/single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg/' --nnet=0 --alarm_level=99.0 --single_exp_plot=3131 --model_idx=5 --reg_model='lin' --wind_hei=5 --wind_wid=5 --spatial_ewma_sigma=30 --spatial_ewma_wind_len=30 --nois_sigma=0.1 --intcp=0 --z_scale=0 --cv_flag=0 --max_steps=50000 --stopping_lag=1000 --activation='sigmoid' --materials_model="non_causal" --training_rounds=1 --n_comp=2 --real_img_path='Data/texture/Octyl_images/003_cropped.tif'         |
||![](Experiments/Octyl_examples/figures/0524203001_sim_reg_no_cv_non_causal_retro/3D_score_clustering_dr_cl_plots_3_sim_real_reg_2d_score_retro_1e-08_loc_info.png)|
|Duel-phase steel | python -uB ./Nonstationarity_Diagnostics/single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet/' --nnet=1 --alarm_level=99.0 --single_exp_plot=3131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=5 --wind_wid=5 --spatial_ewma_sigma=30 --spatial_ewma_wind_len=30 --nois_sigma=0.1 --intcp=0 --z_scale=0 --cv_flag=0 --max_steps=10000 --activation='sigmoid' --materials_model="non_causal" --training_rounds=3 --n_comp=2 --penal_param=0.01 --learning_rate=1e-05 --stopping_lag=1000 --training_batch_size=100 --real_img_path='Data/texture/TwoPhase_images/gr1.jpg'|
||![](Experiments/Dual_phase_examples/figures/0526204001_sim_reg_nnet_no_cv_non_causal_retro/3D_score_clustering_dr_cl_plots_3_sim_real_reg_2d_score_retro_0_01_loc_info.png)|
|Brodatz collage | python -uB ./Nonstationarity_Diagnostics/single_sim_call.py --model_file_folder='${RES_FOLDER_PATH}${IDX}_sim_reg_nnet/' --nnet=1 --alarm_level=99.0 --single_exp_plot=3131 --model_idx=5 --reg_model='nnet_lin' --wind_hei=5 --wind_wid=5 --spatial_ewma_sigma=20 --spatial_ewma_wind_len=20 --nois_sigma=0.1 --intcp=0 --z_scale=0 --cv_flag=0 --max_steps=10000 --activation='sigmoid' --materials_model="non_causal" --training_rounds=3 --n_comp=2 --penal_param=0.01 --learning_rate=1e-05 --stopping_lag=1000 --training_batch_size=100 --real_img_path='Data/texture/TwoPhase_images/gr1.jpg'|
||![](Experiments/Brodatz_examples/0512313002_sim_reg_nnet_no_cv_non_causal_retro/3D_score_clustering_dr_cl_plots_3_sim_real_reg_2d_score_retro_0_01_loc_info.png)|



## Step 2:
The extracted HRs from Step 1 are either classified as one of the previously identified and analyzed MCs in the database or determined to be a new MC not previously cataloged. To do this, the algorithm predicts which existing MC a target HR belongs to and ranks the different MCs based on similarity to the target HR. If the target HR is concluded to be different from any existing MC, with minimal human inspection (only to verify if the target HR is a new MC), the new MC is integrated into the database. This largely shifts the burdens of screening/memorizing morphologies and patterns of MCs from humans to algorithms and thus accelerates discovery/identification of new materials.

To train the model(s) and plot results:

| Data set name         | Commend  |
| :------------------ | :---------------- |
| Kylberg texture data set| python -uB ./Classification/finetune.py --model_name='kylberg_pre_trained' --weights_path='./Classification/bvlc_alexnet.npy' --num_epochs=10 --postfix=postfix --weight_decay=0.0005 |
||![](Experiments/Kylberg_examples/20210529_Kylberg_cla/figures/kylberg_cla_kylberg_pre_trained/AlexNet_seg_res_testing.png)|

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

## Contributing
