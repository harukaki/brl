# brl
reinforcement learning for bridge

## Usage
### Supervised Learning from Wbridge5 datasets
Please download the files "train.txt" and "test.txt" from the following URL and place them in the `your_data_directory`.
https://console.cloud.google.com/storage/browser/openspiel-data/bridge  

Run supervised learning
```bash
python supervised_learning.py iterations=400000 train_batch=128 learning_rate=0.0001 \
eval_every=10000 data_path=your_data_directory save_path=your_model_directory
```

Arguments
```
iterations     Number of epochs
train_batch    Minibatche size
learning_rate  Learning rate for Adam
eval_every     Interval for evaluation and model saving
data_path      Path to the directory where the training dataset is located
save_path      Path to the directory where the trained model will be saved
```
### Reinforcement Learning
Please prepare a baseline model for evaluation.  
For example, it is a model created with the above-mentioned supervised learning.
Alternatively, please use the provided pre-trained model.  

Examples  
Run reinforcement learning without loading initial model.

```bash
python ppo.py EXP_NAME=exp0000 NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
TOTAL_TIMESTEPS=5242880000 UPDATE_EPOCHS=10 LR=0.00001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
VE_COEF=0.5 EVAL_OPP_MODEL_PATH=your_baseline_model_path LOAD_INITIAL_MODEL=False \
LOG_PATH="rl_log" SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```

Run reinforcement learning with loading initial model.  
Please prepare a initial model for the neural network.  
For example, it is a model created with the above-mentioned supervised learning.

```bash
python ppo.py EXP_NAME=exp0001 NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
TOTAL_TIMESTEPS=2621440000 UPDATE_EPOCHS=10 LR=0.000001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
VE_COEF=0.5 EVAL_OPP_MODEL_PATH=your_baseline_model_path LOAD_INITIAL_MODEL=True INITIAL_MODEL_PATH=your_initial_model_path \
LOG_PATH="rl_log" SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```

Aguments
```
EXP_NAME              Name of experiment
# rollout settings
NUM_ENVS              Number of parallels for each actor rollout
NUM_STEPS             Number of steps for each actor rollout
MINIBATCH_SIZE        Minibatch size
TOTAL_TIMESTEPS       Number of steps experienced by the end of the training

# ppo settings
UPDATE_EPOCHS         Number of epochs for ppo update
LR                    Learning rate for Adam
GAMMA　　　　　　　　　　Discount factor gamma
GAE_LAMBDA　　　　　　　GAE lambda
CLIP_EPS              Clip for ppo
ENT_COEF              Entropy coefficient
VF_COEF               Value loss coefficient

# other settings
EVAL_OPP_MODEL_PATH   Path to the baseline model prepared for evaluation
LOAD_INITIAL_MODEL    Whether to load a pretrained model as the initial values for the neural network
INITIAL_MODEL_PATH    Path to the initial model for the neural network
LOG_PATH              Path to the directory where training settings and trained models are saved
SAVE_MODEL　　　　　　  Whether to save the trained model
SAVE_MODEL_INTERVAL   Interval for saving the trained model
```
