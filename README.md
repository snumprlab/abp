# ABP
<a href="https://bhkim94.github.io/projects/ABP/"> <b> Agent with the Big Picture: Perceiving Surroundings for Interactive Instruction Following </b> </a>
<br>
<a href="https://bhkim94.github.io/">Byeonghwi Kim</a>,
<a href="https://www.linkedin.com/in/suvaansh-bhambri-1784bab7/"> Suvaansh Bhambri </a>,
<a href="https://kunalmessi10.github.io/"> Kunal Pratap Singh </a>,
<a href="http://roozbehm.info/"> Roozbeh Mottaghi </a>,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>
<br>
<a href="https://embodied-ai.org/cvpr2021/"> Embodied AI Workshop @ CVPR 2021 </a>

**ABP** (Agent with the Big Picture) is an embodied instruction following agent that exploits surrounding views by additional observations from navigable directions to enlarge the field of view of the agent.

## Installing Dependencies
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

## Downloading Pretrained Model Weights.
Go to ABP directory (this repository)
```bash
export ALFRED_ROOT=$(pwd)
bash download_model.sh
```

## Instruction
The code will be released soon!

## Export results for the [leaderboard](https://leaderboard.allenai.org/alfred/submissions/public)
Run
```bash
python models/eval/leaderboard.py --num_threads 4
```
This will create `.json` file in `exp/pretrained/`. Submit this to the leaderboard

## Citation
```
@inproceedings{kim2021agent,
  author    = {Kim, Byeonghwi and Bhambri, Suvaansh and Singh, Kunal Pratap and Mottaghi, Roozbeh and Choi, Jonghyun},
  title     = {Agent with the Big Picture: Perceiving Surroundings for Interactive Instruction Following},
  booktitle = {Embodied AI Workshop @ CVPR 2021},
  year      = {2021},
}
```
