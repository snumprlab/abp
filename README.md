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

<p align="center">
  🏆 <b>2nd Place</b> 🏆
  <br>
  <a href="https://askforalfred.com/EAI21/">ALFRED Challenge (CVPRW'21)</a>
</p>

**ABP** (Agent with the Big Picture) is an embodied instruction following agent that exploits surrounding views by additional observations from navigable directions to enlarge the field of view of the agent.

## Installing Dependencies
```
conda create -n abp python=3.6 -y
conda activate abp
pip install -r requirements.txt
```
You also need to install [Pytorch](https://pytorch.org/get-started/previous-versions/) depending on your system. e.g ) PyTorch v1.10.0 + cuda 11.1 <br>
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Downloading Pretrained Model Weights.
Go to ABP directory (this repository)
```
export ALFRED_ROOT=$(pwd)
bash download_model.sh
```
Download "Pretrained_Models_FILM" from [this link](https://drive.google.com/file/d/1mkypSblrc0U3k3kGcuPzVOaY1Rt9Lqpa/view?usp=sharing)
```
mv Pretrained_Models_FILM/maskrcnn_alfworld/objects_lr5e-3_005.pth .
mv Pretrained_Models_FILM/maskrcnn_alfworld/receps_lr5e-3_003.pth .
```

## Download
Download the ResNet-18 features and annotation files from <a href="https://huggingface.co/datasets/byeonghwikim/abp_dataset">the Hugging Face repo</a>.
```
git clone https://huggingface.co/datasets/byeonghwikim/abp_dataset data/json_feat_2.1.0
```

## Training
To train ABP, run `train_seq2seq.py` with hyper-parameters below. <br>
```
python models/train/train_seq2seq.py --data <path_to_dataset> --model seq2seq_im_mask --dout <path_to_save_weight> --splits data/splits/oct21.json --gpu --batch <batch_size> --pm_aux_loss_wt <pm_aux_loss_wt_coeff> --subgoal_aux_loss_wt <subgoal_aux_loss_wt_coeff>
```
~~**Note**: As mentioned in the repository of <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>, run with `--preprocess` only once for preprocessed json files.~~ We provide the already preprocessed annotation files and the vocab file. You may not need to run the code with ```--preprocess```. <br>
**Note**: All hyperparameters used for the experiments in the paper are set as default.

For example, if you want train ABP and save the weights for all epochs in "exp/abp" with all hyperparameters used in the experiments in the paper, you may use the command below. <br>
```
python models/train/train_seq2seq.py --dout exp/abp --gpu --save_every_epoch
```
**Note**: The option, `--save_every_epoch`, saves weights for all epochs and therefore could take a lot of space.


## Evaluation
### Task Evaluation
To evaluate ABP, run `eval_seq2seq.py` with hyper-parameters below. <br>
To evaluate a model in the `seen` or `unseen` environment, pass `valid_seen` or `valid_unseen` to `--eval_split`.
```
python models/eval/eval_seq2seq.py --data <path_to_dataset> --model models.model.seq2seq_im_mask --model_path <path_to_weight> --eval_split <eval_split> --gpu --num_threads <thread_num>
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.

If you want to evaluate our pretrained model saved in `exp/pretrained/pretrained.pth` in the `seen` validation, you may use the command below.
```
python models/eval/eval_seq2seq.py --model_path "exp/pretrained/pretrained.pth" --eval_split valid_seen --gpu --num_threads 4
```

## Export results for the [leaderboard](https://leaderboard.allenai.org/alfred/submissions/public)
Run
```
python models/eval/leaderboard.py --num_threads 4
```
This will create `.json` file in `exp/pretrained/`. Submit this to the leaderboard

## Expected Result
| Test Seen SR   | Test Seen GC    | Test Unseen SR  | Test Unseen GC  |
| -------------- | --------------- | --------------- | --------------- |
| 54.21% (41.62) | 60.03% (48.22%) | 26.16% (16.68%) | 36.42% (25.69%) |

## Citation
```
@inproceedings{kim2021agent,
  author    = {Kim, Byeonghwi and Bhambri, Suvaansh and Singh, Kunal Pratap and Mottaghi, Roozbeh and Choi, Jonghyun},
  title     = {Agent with the Big Picture: Perceiving Surroundings for Interactive Instruction Following},
  booktitle = {Embodied AI Workshop @ CVPR 2021},
  year      = {2021},
}
```
