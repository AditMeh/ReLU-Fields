#!/bin/bash 

#SBATCH -A rahul
#SBATCH -q rahul 
#SBATCH -p rahul

#SBATCH -c 4
#SBATCH -w voyager
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/pkgs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/pkgs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/pkgs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/pkgs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
cd /voyager/projects/aditya/ReLU-Fields/

conda activate 3D
python main.py --config_path $1
