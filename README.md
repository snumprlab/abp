# ABP++
ABP++



# Downloading Pretrained Model Weights.
Go to ABP directory (this repository)
```bash
export ALFRED_ROOT=$(pwd)
bash download_model.sh
```

# Installing Dependencies
```bash
conda create -n abp python=3.6 -y
conda activate abp
pip install -r requirements.txt
```
You also need to install Pytorch depending on your system. e.g ) PyTorch v1.10.0 + cuda 11.1 <br>
Refer [here](https://pytorch.kr/get-started/previous-versions/)
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# Export results for the [leaderboard](https://leaderboard.allenai.org/alfred/submissions/public)
Run
```bash
python models/eval/leaderboard.py --num_threads 4
```
This will create `.json` file in `exp/pretrained/`. Submit this to the leaderboard
