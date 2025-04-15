#!/bin/bash

# install and activate conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x ~/Miniconda3-latest-Linux-aarch64.sh
source ~/miniconda3/bin/activate

# navigate to final
cd ~/Downloads/final

# create and activate reflection env
conda env create -f environment.yml
conda activate reflection

# done
echo "Conda environment 'reflection' created and activated."