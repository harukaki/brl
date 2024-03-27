# brl
reinforcement learning for bridge

## Installation
Please install the necessary packages according to the `requirements.txt`.  
Note that you need to install the appropriate versions of jax and jaxlib according to your execution environment.  
Additionally, we are using pgx as the environment for bridge, and currently, we support version 1.4.0 of [pgx](https://github.com/sotetsuk/pgx). 
```bash
pip install -r requirements.txt
```
For bridge bidding in pgx, downloading the Double Dummy Solver (DDS) dataset is required. Please download the DDS dataset according to [pgx bridge bidding documentation](https://github.com/sotetsuk/pgx/blob/main/docs/bridge_bidding.md).
```py
from pgx.bridge_bidding import download_dds_results
download_dds_results()
```

## Pre-trained models
Parameters trained by this repository are published. 
| Model                             | Description                                     | Score against wbridge5 |
|-----------------------------------|-------------------------------------------------|------------------------|
| model-sl.pkl                      | Supervised Learning from wbridge5               | -0.56 IMPs/b           |
| model-from-scrach-rl.pkl          | Reinforcement Rearning from scrach              | -0.64 IMPs/b           |
| model-pretrained-rl.pkl           | RL after SL pretraining                         |  0.88 IMPs/b           |
| model-pretrained-rl-with-fsp.pkl  | RL after SL pretraining with FSP                |  1.24 IMPs/b           |
| model-pretrained-rl-with-pfsp.pkl | RL after SL pretraining with mix of SP and PFSP |  0.89 IMPs/b           |

For more details on each training, please refer to [`bridge_models/README`](https://github.com/harukaki/brl/tree/main/bridge_models).
## Evaluation models
To evaluate pre-trained models against each other, please use the following command:  
Example
```bash
python eval.py team1_model_path=bridge_models/model-pretrained-rl.pkl \
  team2_model_path=bridge_models/model-sl.pkl num_eval_envs=100
```

Here's an example of the output:
```
Loading dds results from dds_results/test_000.npy ...
num envs: 100
---------------------------------------------------
bridge_models/model-pretrained-rl.pkl vs. bridge_models/model-pretrained-rl.pkl
IMP: 0.47999998927116394 ± 0.5320970416069031
```

## Supervised Learning from Wbridge5 datasets
Please download the "train.txt" and "test.txt" files, which are part of the dataset published by Openspiel, from the specified URL.  
After downloading, place these files in your `your_data_directory`.  
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/bridge_supervised_learning.py

Example  

Run supervised learning
```bash
python sl.py iterations=400000 train_batch=128 learning_rate=0.0001 \
  eval_every=10000 data_path=your_data_directory save_path=your_model_directory
```


## Reinforcement Learning
Please prepare a baseline model for evaluation and enter its file path in `eval_opp_model_path`.  
For instance, the pre-trained model provided through supervised learning.  

Examples  
  
Run reinforcement learning without loading initial model.

```bash
python ppo.py num_envs=8192 num_steps=32 minibatch_size=1024 \
  total_timesteps=5242880000 update_epochs=10 lr=0.00001 gamma=1 gae_lambda=0.95 ent_coef=0.001 \
  VE_COEF=0.5 num_eval_envs=100 eval_opp_model_path="bridge_models/model-sl.pkl" num_eval_step=10 \
  load_initial_model=False log_path="rl_log" exp_name=exp0000 save_model=True save_model_interval=100
```

Run reinforcement learning with loading initial model.  
Please prepare a initial model for the neural network and enter its file path in `initial_model_path`.  
For instance, the pre-trained model provided through supervised learning. 

```bash
python ppo.py num_envs=8192 num_steps=32 minibatch_size=1024 \
  total_timesteps=2621440000 update_epochs=10 lr=0.000001 gamma=1 gae_lambda=0.95 ent_coef=0.001 \
  VE_COEF=0.5 num_eval_envs=100 eval_opp_model_path="bridge_models/model-sl.pkl" num_eval_step=10 \
  load_initial_model=True initial_model_path="bridge_models/model-sl.pkl" \
  log_path="rl_log" exp_name=exp0001 save_model=True save_model_interval=100
```

## Evaluation with wbridge5
You can use the  [`bridge_env`](https://github.com/yotaroy/bridge_env) submodule to play a network match against the rule-based bridge AI, [Wbridge5](http://www.wbridge5.com/), on localhost. Please note that Wbridge5 only runs on Windows.

Install the `bridge_env` submodule
```bash
git submodule update --init --recursive
cd submodule/bridge_env
python setup.py install
cd ../
```
Execute a network match between a trained model and Wbridge5.  
Example
```bash
bash eval_wb5.sh bridge_models/model-sl.pkl relu DeepMind log_wb5 2000 2001
```

Launch Wbridge5, set "localhost" as the server, connect the positions of "N" and "S" to the first port, and connect the positions of "E" and "W" to the second port.  
