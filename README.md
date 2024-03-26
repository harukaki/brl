# brl
reinforcement learning for bridge
## Pre-trained models
Parameters trained by this repository are published. 
| Model                             | Description                                     | Score against wbridge5 |
|-----------------------------------|-------------------------------------------------|------------------------|
| model-sl.pkl                      | Supervised Learning from wbridge5               | -0.56 IMPs/b           |
| model-from-scrach-rl.pkl          | Reinforcement Rearning from scrach              | -0.64 IMPs/b           |
| model-pretrained-rl.pkl           | RL after SL pretraining                         |  0.88 IMPs/b           |
| model-pretrained-rl-with-fsp.pkl  | RL after SL pretraining with FSP                |  1.24 IMPs/b           |
| model-pretrained-rl-with-pfsp.pkl | RL after SL pretraining with mix of SP and PFSP |  0.89 IMPs/b           |

For more details on each training, please refer to `bridge_models/README`.
## Usage
### 1. Installation
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

### 2. Supervised Learning from Wbridge5 datasets
Please download the "train.txt" and "test.txt" files, which are part of the dataset published by Openspiel, from the specified URL.  
After downloading, place these files in your `your_data_directory`.  
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/bridge_supervised_learning.py

Example  

Run supervised learning
```bash
python sl.py iterations=400000 train_batch=128 learning_rate=0.0001 \
eval_every=10000 data_path=your_data_directory save_path=your_model_directory
```


### 3. Reinforcement Learning
Please prepare a baseline model for evaluation and enter its file path in `your_baseline_model_path`.  
For example, it is a model created with the above-mentioned supervised learning. 

Examples  
  
Run reinforcement learning without loading initial model.

```bash
python ppo.py NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
TOTAL_TIMESTEPS=5242880000 UPDATE_EPOCHS=10 LR=0.00001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
VE_COEF=0.5 NUM_EVAL_ENVS=100 EVAL_OPP_MODEL_PATH=your_baseline_model_path NUM_EVAL_STEP=10 \
LOAD_INITIAL_MODEL=False LOG_PATH="rl_log" EXP_NAME=exp0000 SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```

Run reinforcement learning with loading initial model.  
Please prepare a initial model for the neural network and enter its file path in `your_initial_model_path`.  
For example, it is a model created with the above-mentioned supervised learning.

```bash
python ppo.py NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
TOTAL_TIMESTEPS=2621440000 UPDATE_EPOCHS=10 LR=0.000001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
VE_COEF=0.5 NUM_EVAL_ENVS=100 EVAL_OPP_MODEL_PATH=your_baseline_model_path NUM_EVAL_STEP=10 \
LOAD_INITIAL_MODEL=True INITIAL_MODEL_PATH=your_initial_model_path \
LOG_PATH="rl_log" EXP_NAME=exp0001 SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```
