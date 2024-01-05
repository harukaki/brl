# brl
reinforcement learning for bridge

## Usage
### Supervised Learning from Wbridge5 datasets
Please download the files "train.txt" and "test.txt" from the following URL and place them in the data_path directory.
https://console.cloud.google.com/storage/browser/openspiel-data/bridge  

run supervised learning
```bash
python sl_deepmind.py --data_path DATA_PATH [--save_path SAVE_PATH]
```

Arguments
```bash
--data path    Path to the directory where the training dataset is located
--save_path    Path to the directory where the trained model will be saved
```
### Reinforcement Learning
```bash
python new_ppo.py
```
```example
python new_ppo.py NUM_ENVS=10 NUM_STEPS=5 TOTAL_TIMESTEPS=1000 UPDATE_EPOCHS=2 MINIBATCH_SIZE=10 NUM_EVAL_ENVS=10 TRACK=False
```

