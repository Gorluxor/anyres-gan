export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# A100 40GB clusters
module purge
module load gcc
module unload zlib
module load cudnn
module unload cuda/10.2.89
module unload libiconv/1.16
module unload xz
module unload zlib
module unload libxml2
module load cuda/11.5.0

# A100 80GB clusters
# module purge
# module load cudnn
echo "CUDA_HOME: $CUDA_HOME"

source activate anyres-gan

rm -rf ~/.cache/torch_extensions
# --metrics fid2k_full,pfid2k removed, using fid2k_base (which reconfigures to 1k)

# TEST RUN without metrics
# run: stage 2 patch training
# --mbstd-group 4 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --cfg=stylegan3-r --gpus=4 --batch=32 --gamma=2 \
	--mirror=1 --aug=noaug --kimg 200 --snap 1  --batch-gpu 4 \
	--outdir training-runs/SS \
    --data /../SS/LR \
	--training_mode=patch --g_size 1024 --random_crop=True \
	--teacher ./pretrained/stylegan3-r-ffhq-1024x1024.pkl \
	--data_hr /../SS/HR \
	--metrics fid2k_base --teacher_lambda 12 --teacher_mode crop --scale_max 1 --scale_min 1 \
	--scale_anneal -1 --scale_mapping_min 1 --scale_mapping_max 1 --patch_crop=True \
    --use_hr=True --use_scale_on_top=False --actual_res 4096 --ds_mode bicubic --base_probability 2\
	--l2_lambda 0 

# For now scale mapping is disabled, as well as mapping 
# Actual res is always 4096 for our mode, when used with use_hr=True
# ds_mode bicubic|nearest|bilinear 
# base probability is >1 so that we disable 1k training
# current loss.py is using teacher_mode => crop

# variable parameters
# teacher_lambda 0 ... 12 etc
# l2_lambda => weight decay