# brl
reinforcement learning for bridge

## Usage
### Supervised Learning from Wbridge5 datasets
Please download the files "train.txt" and "test.txt" from the following URL and place them in the `your_data_directory`.
https://console.cloud.google.com/storage/browser/openspiel-data/bridge  

run supervised learning
```bash
python supervised_learning.py data_path=your_data_directory [save_path=your_model_directory]
```

Arguments
```bash
data_path    Path to the directory where the training dataset is located

save_path    Path to the directory where the trained model will be saved
```
### Reinforcement Learning
Please prepare a baseline model for evaluation.  
For example, conduct supervised learning and save the trained model.
Alternatively, please use the provided pre-trained model.
```bash
python ppo.py eval_opp_model_path=EVAL_OPP_MODEL_PATH
```
Arguments

example
```bash
python ppo.py NUM_ENVS=10 NUM_STEPS=5 TOTAL_TIMESTEPS=1000 UPDATE_EPOCHS=2 \
 MINIBATCH_SIZE=10 NUM_EVAL_ENVS=10 TRACK=False
```

```
NUM_ENVS
  
NUM_STEPS

TOTAL_TIMESTEPS

UPDATE_EPOCHS

MINIBATCH_SIZE

NUM_EVAL_ENVS

TRACK
```
