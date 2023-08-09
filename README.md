# Training
This are training examples for each model

## MSA (Medical Sam Adapter)

### Training from scratch
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../datasets/Lits/imgs_1ch/liver_only/ -in_channels 1 -b 4

### Fine Tuning
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../datasets/Lits/imgs_1ch/liver_only/ -in_channels 1 -b 4 -weights ./checkpoint/checkpoint_best.pth

Note: log file will be saved on the -weights experiment log folder. Checkpoints are saved in the actual experiment checkpoint folder.

### Continuing Training
python train.py -net sam -mod sam_adpt -exp_name Lits2017_Liver_Only -image_size 512 -dataset lits -data_path ../datasets/Lits/imgs_1ch/liver_only/ -in_channels 1 -b 4 -epoch_ini 40 -weights ./checkpoint/checkpoint_best.pth

# HybridGNet

### Training from scratch
python3 trainerHybrid.py --name HybridGNet --inputsize 512 --extended --epochs 100

Note: Batch size is 4 by default. This can me modified in line 329 of trainerHybrid.py.
Note: extended uses the whole data, if this is not added, the model will be trained on the split 1, consisting of 50% random-selected data subset.

--load
