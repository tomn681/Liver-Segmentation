conda info --envs
nvidia-smi
exit
passwd
exit
ls
cd Code
mkdir Code
mkdir Data
ls Data
cd Data
cd ../Code
git clone https://github.com/joaquin2c/MedSegDiff.git
ls
cd ../Data
exit
cd Data
ls
ls -l
ls
rm liver_only.zip
ls
wget https://drive.google.com/file/d/12IiyDnnYz1uRYO3CBU90IYHmK_32ra1g/view
ls
rm view
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12IiyDnnYz1uRYO3CBU90IYHmK_32ra1g' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12IiyDnnYz1uRYO3CBU90IYHmK_32ra1g" -O liver.zip && rm -rf /tmp/cookies.txt
exit
ls
cd Code
ls
cd MedSegDiff
ls
cd ../../Data
ls
unzip liver.zip
apt install unzip
sudo apt install unzip
conda create --name experiments
conda create --name "experiments"
conda env create --name experiments
pip -v
pip list
pip install conda
conda -v
conda --version
conda create --name env
tmux new
gunzip -v
ls
gzip -d liver.zip
gzip -df liver.zip
gzip --df liver.zip
gzip -f liver.zip
ls
gzip liver.zip
gzip --decompress liver.zip
gunzip --decompress liver.zip
tar -v
tar -xvzf liver.zip
tar -xvzf "liver.zip"
ls -l
ls -l --block-size=M
unzip -v
sudo apt install unzip
tar -x liver.zip
tar -xf liver.zip
cat unzip.py
nano unzip.py
python unzip.py
python3 unzip.py
nano unzip.py
python3 unzip.py
ls
cd liver_only
ls
cd images
ls
..
ls
cd ..
ls
ls -l
cd full_data
ls
ls |wc-l
ls -1|wc-l
ls | wc-l
ls -l | wc
ls | wc -l
ls images | wc -l
cd ..
ls
cd split_1
ls
ls images | wc -l
cd ..
cd full_data
ls
ls val | wc -l
ls val/images | wc -l
cd ..
tmux attach
cd ../Code/MedSegDiff
ls ../../Data
ls ../../Data/liver_only
ls ../../Data/liver_only/full_data
tmux attach
conda create --name env
pip install -r requirement.txt
sudo conda create --name env
pip install pytorch==1.8
pip install pytorch=1.8
pip install pytorch>=1.8
pip install torch>=1.8
pip install torch==1.8
pip install torch==1.11.0
pip install numpy
pip install pandas
pip install blobfile
pip install nilabel
pip install nibabel
pip install opencv-python
pip install scikit-image
pip install matplotlib
pip install bachgenerators
pip install batchgenerators
pip install visdom
pip install torchsummary
pip install albumentations
tmux attach
pip install torchvision
nvidai-smi
nvidia-smi
htop
pip install torchvision
cd ../../Data
ls
rm liver.zip
ls
cd ../Code/MedSegDiff
ls
pip install torchvision
tmux attach
ls
cd scripts
ls
cd ../guided_diffusion
ls
nano train_util.py
cd ..
ls
cd scripts
ls
nano segmentation_train.py
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
ls
cd ..
ls
cd guided_diffusion
ls
nano train_util.py
tmux attach
nvidia-smi
tmux attach
cd ..
ls
ls ../../Data
ls ../../Data/liver_only
ls ../../Data/liver_only/split_1
cd guided_diffusion
nano train_util.py
tmux attach
exit
tmux attach
exit
tmux attach
nvidia-smi
df -H
exit
nvidia-smi
tmux attach
exit
tmux attach
exit
cd ..
cd Code/MedSegDiff
ls
pip install -r requirement.txt
conda create -n exp
conda create --name exp
pip install blobfile
pip install batchgenerators
pip install visdom
python scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 20000 --log_interval 2000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 10
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 20000 --log_interval 2000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 10
condainfo --envs
conda info --envs
conda create --name exp
conda create --name {exp}
conda create --n exp python==3.7
conda create -n exp python==3.7
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 20000 --log_interval 2000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 10
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 20000 --log_interval 2000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 20
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 20000 --log_interval 2000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 10000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/split_1"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 10000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
ls
cd Code
ls
cd MedSegDiff
ls
git status
ls
cd guided_diffusion
ls
cd ..
ls
cd guided_diffusion
ls
nano train_util.py
nano liverloader.py
cd ..
ls
git status
nano .gitignore
git status
ls
cd results
ls
nano progress.csv
du -sh
df -H
ls
cd ..
ls
cd scripts
ls
nano segmentation_train.py
git status
cd =1.8
cd /=1.8
ls
git add .
git commit -am "correct format"
git config --global user.email "j.curimil.c@gmail.com"
git config --global user.name "joaquin2c"
git commit -am "correct format"
git push
nano segmentation_train.py
git add .
git commit -am "correct format"
git push
cd ..
ls
rm -r results
ls
tmux attach
tmux new
ls
cd guided_diffusion
ls
cd ..
cd guided_diffusion
nano train_util.py
ls
tmux attach
cd ..
ls ../../Data
ls
ls ../../Data/liver_only
ls ../../Data/liver_only/full_data
ls ../../Data/liver_only/full_data/images
ls ../../Data/liver_only/images
ls ../../Data/liver_only/images -l | w -c
ls ../../Data/liver_only/images -1 | w -c
ls ../../Data/liver_only/images -1 | wc -l
ls ../../Data/liver_only/full_data/images -1 | wc -l
ls ../../Data/liver_only/full_data
ls ../../Data/liver_only/full_data -1 | wc -l
ls ../../Data/liver_only/full_data/val -1 | wc -l
ls ../../Data/liver_only/full_data/val/images -1 | wc -l
ls ../../Data/liver_only/val -1 | wc -l
ls ../../Data/liver_only
ls ../../Data/liver_only/full_data
tmux attach
ls ../../Data/liver_only/full_data
tmux attach
ls
cd guided_diffusion
lñs
ls
nano liverloader.py
tmux attach
nvidia-smi
tmux attach
exit
cd Data
ls
cd liver_only
ls
ls split_1 -1 | wc -l
ls split_1/images -1 | wc -l
tmux attach
ls
cd ../../Code
ls
cd MedSegDiff
ls
cd guided_diffusion
ls
cd train_util.py
nano train_util.py
tmux attach
l
cd ..
ls
cd results
ls
tmux attach
nvidia-smi
cd ..
tmux attach
exit
tmux attach
exit
tmux attach
exit
tmux attach
exit
nvidia-smi
tmux attach
exit
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 30000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
cd ..
exit
nvidia-smi
tmux attach
cd Code
ls
cd MedSegDiff
ls
cd results
ls
nano progres.csv
nano progress.csv
ls
cd Code
ls
git status
mv MedSegDiff ..
ls
ls ..
git clone https://github.com/tomn681/Liver-Segmentation.git
ls
git status
cd ..
ls
mv MedSegDiff Code
ls
cd Code
ls
cd Liver-Segmentation
ls
git branch -a
git branch
git checkout remotes/origin/Joaquin_main
git branch -a
git status
git checkout main
git branch
git fetch
git branch
git branch -a
git checkout Joaquin_main
git branch
git branch -a
git status
git pull --all
git branch
git branch -a
cd
cd ~/Code/Liver-Segmentation
ls
git merge main
ls
git status
git commit -am "merge branch"
hit add .
git add .
git commit -am "merge branch"
git push
ls
cd Medical-SAM-Adapter
ls
cd dataset
nano dataset.py
l
cd train.py
nano train.py
tmux attach
tmux new
git branch
git status
git add .
git commit -am "new trans and threshold"
git push
tmux attach
pytorch
python3
ls
git branch
ls ../../..
ls ../../../Data
ls ../../../Data/liver_only
ls ../../../Data/liver_only/full_data
ls ../../../Data/liver_only
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 b 4
python3 train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 b 4
ls
conda env create -f environment.yml
conda info --envs
conda activate sam_adapt
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 b 4
pip install tensorboardX
pip install tensorboardX==2.2
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 b 4
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
pip install torch==1.12.1+cu113
pytorch -v
pyton -v
python -v
python --version
conda deactivate
ls
nano environment.yml
conda env remove --name sam_adapt
conda create --name sam_adapt
nano environment.yml
nvcc --version
nvidia-smi
python
python3
nano environment.yml
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
conda env create -f environment.yml
conda update -n base -c defaults conda
conda activate sam_adapt
conda update -n base -c defaults conda
conda deactivate
nano environment.yml
conda env create -f environment.yml
conda env remove --name sam_adapt
conda env create -f environment.yml
nano environment.yml
conda env create -f environment.yml
nano environment.yml
conda env create -f environment.yml
pip install tomli==2.0.1
pip install torch==1.12.1+cu113
pip install torch==1.12.0+cu113
pip install torch==1.12.1
df -H
ls
cd ..
ls
cd MedSegDiff
ls
de results
cd results
ls
rm savedmodel_000000.pt
rm emasavedmodel_0.9999_000000.pt
rm emasavedmodel_0.9999_030000.pt
ls -l
rm optsavedmodel000000.pt
ls
rm optsavedmodel030000.pt
ls
rm savedmodel_030000.pt
df -H
-h
du -h
cd ..
ls
du -h
ls
cd Data
du -h
cd ../Code
du -h
conda env --list
conda list
conda info --envs
conda env remove --name sam_adapt
df -H
conda env create -f environment.yml
ls
cd MedSegDiff
cd ..
cd Liver-Segmentation
ls
cd Medical*
ls
conda env create -f environment.yml
df -H
conda activate
conda activate sam_adapt
pip install torch==1.12.1+cu113
pip install torch==1.12.1
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
pip install tensorboardX
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
pip install einops
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
pip install torch-2.0.1
pip install torch==2.0.1
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
conda deactivate
conda env remove --name sam_adapt
nano environment.yml
conda env create -f environment.yml
conda activate sam_adapt
pip install torchvision==0.13.1
conda deactivate
conda env remove --name sam_adapt
nano environment.yml
conda activate sam_adapt
conda env create -f environment.yml
df -H
conda env remove --name sam_adapt
df -H
cd ..
du -h
ls
cd Code
du -h
cd ..
cd Data
du -h
cd ..
cd Code
du -h
cd ..
cd pooch
du -h
cd .
cd 
cd
cd ..
ls
cd tsotta
du -h
du -h tsotta
ls
cd jcurimil
ls
cd Code
ls
cd MedSegDiff
ls
cd results
ls
cd ..
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install blobfile
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install nibabel
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install torchvision
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
conda deactivate
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install torch==2.0.1
python
python3
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install torchvision torchaudio
df -H
ls
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install opencv-python
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install tifffile
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
pip install threadpoolctl
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
df -H
pip install websocket
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 60000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
df -H
python3 scripts/segmentation_train.py --data_name LIVER --data_dir "../../Data/liver_only/full_data"  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --save_interval 90000 --log_interval 1000 --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 18
ls
rm -r results
ls
df -H
dh
du -H
cd ..
du -s
du -sh
cd ..
du -sh
ls
cd Data
ls
cd liver_only
ls
rm -r split_1
rm -r split_2
df -H
cd full_data
du -sh
cd ..
du -sh
cd ../../Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
nano environment.yml
conda env create -f environment.yml
conda activate sam_adapt
df -H
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
ls ../../../Data/liver_only/
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
python
conda list -f pytorch
pip install torch==2.0.1
pip install torchvision
pip install torch==2.0.1
pip install torchvision==0.15.1
python
nano environment.yml
conda deactivate
conda env remove --name sam_adapt
conda env create -f environment.yml
python
conda activate sam_adapt
python
pip install torch==2.0.1
ls
python
conda list -f TensorRT
ls
dc data
conda deactivate
conda env remove --name sam_adapt
nano environment.yml
conda env create -f environment.yml
conda activate sam_adapt
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
conda install conda=23.7.2
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
conda deactivate
conda env remove --name sam_adapt
nano environment.yml
conda env create -f environment.yml
conda activate
conda activate sam_adapt
pip install torch==1.13.1+cu117
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
nano environment.yml
pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
pip installpip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install tensorboardX
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
pip install open-cv
conda deactivate
conda env remove --name sam_adapt
nano environment.yml
conda env create -f environment.yml
pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
df -H
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch==1.13.1 torchvision==0.15.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==1.13.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python
df -H
conda activate sam_adapt
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
ls
conda uninstall torchvision
df -H
pip install torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchvision==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 4
exit
tmux attach
python3
pip instal torch==2.0.1
pip install torch==2.0.1
python3
tmux attach
cd Code
ls
cd MedSegDiff
ls
cd results
ls
df -H
tmux attach
df -H
tmux attach
ls
tmux attach
df -H
conda info --envs
nvidia-smi
tmux attach
cd ../../..
cd Data
ls
cd liver_only
ls
rm -r full_data
tmux attach
python3
tmux attach
exit
ls
cd Code
ls
cd liver-Segmentation
cd Liver-Segmentation
ls
git branch
rm -r MedSegDiff
ls
cd ..
ls
mv MedSegDiff Liver-Segmentation
ls
cd Liver-Segmentation
ls
git status
git add .
cd MedSegDiff
ls
cd scripts
ls
cd segmentation_sample.py
nano segmentation_sample.py
cd ..
cd guided_diffusion
ls
nano gaussian_diffusion.py
nano dpm_solver.py
cd ..
git status
git add .
git commit -am "MedSegDiff Modifications"
git push
git status
git branch
git checkout Joaquin_main
git branch -a
git branch Joaquin_main
git branch
git checkout Joaquin_main
git branch
git status
ls
cd ..
ls
mv MedSegDiff ..
ls
cd ..
cp MedSegDiff Liver-Segmentaion/ 
ls
cd Liver_Segmentation
cd Liver-Segmentation
ls
mkdir new
cd ..
cp -r MedSegDiff Liver-Segmentaion/new/
ls
cp -r MedSegDiff Liver-Segmentaion/new
cp -r MedSegDiff Liver-Segmentation/new
ls
cd Liver-Segmentation
ls
cd new
ls
mv MedSegDiff ..
ls
cd ..
rm -r new
ls
cd MedSegDiff
ls
git status
rm -fr .git
ls
git status
git branch
git branch -a
cd ..
git status
git add .
git commit -am "MedSegDiff Modified"
git branch}
git branch
git push
ls
cd Medical-SAM-Adapter
ls
nano environment.yml
cd ..
ls
df -H
conda envs --info
conda env --info
conda info --envs
conda env remove --name sam_adapt
df -H
python3
pip install torch torchvision
python3
df -H
ls
cd UNet++
conda create -n=<env_name> python=3.6 anaconda
conda create -n=<Unet> python=3.6 anaconda
conda create -n=Unet python=3.6 anaconda
conda list --show-channel-urls
conda create -n=Unet python=3.6 anaconda
~/.bashrc
conda config --remove-key channels
ls
nano train.py
ls ../../
ls ../../../Data
ls ../../../Data/liver_only
ñs
ls
python train.py --dataset ../../../Data/liver_only --arch NestedUNet
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet -b 16
tmux new
tmux attach
nano dataset.py
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
nvidia-smi
tmux attach
nvidia-sm
nvidia-smi
tmux attach
exit
nvidia-smi
tmux attach
exit
df -H
ls
cd Code
ls
cd Data
ls
cd liver_only_NestedUNet_woDS
ls
ls -l
ls -lh
nano log.csv
exit
nvidia-smi
tmux attach
exit
tmux attach
exit
tmux attach
cd Code
ls
cd Data
ls
cd liver_only_NestedUNet_woDS
ls
ls -l
ls -lh
nvidia-smi
exit
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet -b 16
pip install opencv-python
df -H
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet -b 16
pip install threadpoolctl
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet -b 16
pip install tqdm
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet -b 16
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet -b 16 --input_channels 1
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet -b 18 --input_channels 1
python3 train.py --dataset ../../../Data/liver_only --arch NestedUNet --epochs 300 -b 18 --input_channels 1
exit
nvidia-smi
tmux attach
exit
ls
cd Data
ls
cd liver_only
ls
df -H
ls -l images
ls -l
ls -1 | wc-l
ls -1 | wc -l
ls -1 | wc -l images
cd images
ls -1 | wc -l 
cd ../../..
cd Cod
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
git status
nano environment.yml
ls ../../..
ls ../../../Data
ls ../../../Data/liver_only
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 18
conda env create -f environment.yml
nano environment.yml
conda env create -f environment.yml
conda remove --name sam_adapt
conda remove --name sam_adapt --all
conda env create -f environment.yml
conda activate sam_adapt
python
pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python
pip install torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 18
tmux new
ls
tmux new
tmux attach
ls
nano function.py
tmux attach
nvidia-smi
tmux attach
nvidia-smi
exit
nvidia-smi
tmux attach
python
tmux attach
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
cd logs
ls
cd Lits2017_Liver_Only_2023_08_23_00_08_32
ls
cd Log
ls
cd ..
cd Model
ls
cd Samples
ls
cd ..
cd Samples
cd ..
ls
cd Samples
ls
cd ..
ls
cd Log
ls
cd ..
ls
cd ..
s
ls
cd ..
ls
cd runs
ls
cd sam
ls
ls -l
cd 2023-08-23T00:08:30.367344
ls
ls -l
nvidia-smi
cd ..
lcd ..
cd ..
ls
nano train.py
tmux attach
ls
cd figs
ls
cd ..
ls
cd checkpoint
ls
cd sam
ls
cd 2023-08-23T00:08:30.367344
ls
cd ..
ls
cd logs
ls
cd Lits2017_Liver_Only_2023_08_23_00_08_32
ls
cd Log
ls
cd Model
ls
cd ..
cd Model
ls
du -H
du
df
du -h
ls
ls -s
ls -l
ls -lh
tmux new
cd ../..
ls
cd ..
ls
cd data
ls
cd isisv
cd isic
ls
ls -l
cd ..
ld
ls
cd ..
ls
cd models
ls
cd ..
ls
cd runs
ls
cd sam
ls
ls -l
cd 2023-08-23T00:08:30.367344
ls
ls -l
tmux attach
cd ..
ls
cd logs
ls
cd Lits2017_Liver_Only_2023_08_23_00_08_32
ls
tmux attach
ls
cd Log
ls
ls -l
tmux attach
la
cd ..
ls
cd ..
ls
cd checkpoint
ls
cd sam
ls
cd 2023-08-23T00:08:30.367344
ls
ls -l
cd ..
ls
cd ..
ls
cd train
nano train.py
tmux attach
ls
tmux attach
cd logs
ls
tmux attach
cd ..
mkdir log
ls
tmux attach
rm -r log
mkdir logcsv
tmux attach
ls
cd logs
ls
tmux attach
exit
tmux attach
conda activate sam_adp
conda info --envs
conda activate sam_adapt
python
tmux attach
ls
cd Code
ls
cd Liver-Segmentation
ls
Medical-SAM-Adapter
cd Medical-SAM-Adapter
ls
nano function.py
nano train.py
tmux attach
nano function.py
tmux attach
nano function.py
tmux attach
nano function.py
nano train.py
tmux attach
ls
cd cfg.py
nano cfg.py
cd conf
ls
nano global_settings.py
tmux attach
exit
nano funcion.py
nano function.py
exit
tmux attach
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
tmux attach
nano train.py
tmux attach
nano train.py
tmux attach
nano train.py
tmux attach
nano train.py
nano function.py
tmux attach
nano function.py
tmux attach
nano train.py
nano function.py
tmux attach
nano function.py
tmux attach
tmux new
tmux attach
nano function.py
nano train.py
tmux attach
nano function.py
tmux attach
ls
cd logcsv
ls
nano log_V2.csv
cd ..
ls
nano train.py
nvidia-smi
exit
tmux attach
exit
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
l
cd logcsv
ls
nano log_V2.csv
exit
tmux attach
exit
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 18
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 36
ls
nano train.py
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 36
nano train.py
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 36
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only_V1 -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 36
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only_V2 -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 36
tmux attach
ls
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
cd logcsv
ls
cd ..
ls
cd models
ls
cd ..
cd runs
ls
cd sam
ls
ls -l
cd 2023-08-23T04:45:08.891485
ls
cd ..
ls
cd ..
ls
cd ..
ls
cd logs
ls
cd Lits2017_Liver_Only_V2_2023_08_23_04_45_10
ls
cd Model
ls
ls -l
ls
ls -l
tmux attach
ls
ls -l
tmux attach
tmux new
ls
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
cd logcsv
ls
mv log_V2.csv log_V2_pt1.csv
ls
cd ..
ls
nano train.py
ls
tmux attach
exit
tmux attach
exit
ls
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
cd logs
ls
cd -l
ls -l
cd Lits2017_Liver_Only_V2_2023_08_23_04_45_10
ls
cd Model
ls
cd ..
ls
ls -l
mv Lits2017_Liver_Only_V2_2023_08_23_04_45_10 Lits2017_LO_pt1
ls
cd Lits2017_LO_pt1
ls
cd Model
ls
ls -l
cd ../../..
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only_V2_pt2 -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 36 -weights logs/Lits2017_LO_pt1/Model/checkpoint_best.pth
conda activate sam_adapt
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only_V2_pt2 -image_size 512 -dataset lits -data_path ../../../Data/liver_only/ -in_channels 1 -b 36 -weights logs/Lits2017_LO_pt1/Model/checkpoint_best.pth
exit
cd Code
ls
cd Liver-Segmentation
ls
cd Medical-SAM-Adapter
ls
cd logcsv
ls
ls -l
nano log_V2_pt2.csv
tmux attach
exit
