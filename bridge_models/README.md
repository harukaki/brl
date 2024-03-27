# Details of each model's training
## `model-sl.pkl`
The WBridge5 supervised learning model.  
Training settings can be configured with the following command:
```bash
python sl.py iterations=400000 train_batch=128 learning_rate=0.0001 \
  eval_every=10000 data_path=your_data_directory save_path=your_model_directory
```
## `model-from-scrach-rl.pkl`
Initialized with random parameters and trained using PPO through a simple self-play method.  
Training settings can be configured with the following command:
```bash
python ppo.py NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
  TOTAL_TIMESTEPS=5242880000 UPDATE_EPOCHS=10 LR=0.00001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
  VE_COEF=0.5 NUM_EVAL_ENVS=100 EVAL_OPP_MODEL_PATH="bridge_models/model-sl.pkl" NUM_EVAL_STEP=10 \
  LOAD_INITIAL_MODEL=False LOG_PATH="rl_log" EXP_NAME=from-scrach-rl SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```
## `model-pretrained-rl.pkl`
Initialized with the WBridge5 supervised learning model and trained using PPO with a simple self-play method.  
Training settings can be configured with the following command:
```bash
python ppo.py NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
  TOTAL_TIMESTEPS=2621440000 UPDATE_EPOCHS=10 LR=0.000001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
  VE_COEF=0.5 NUM_EVAL_ENVS=100 EVAL_OPP_MODEL_PATH="bridge_models/model-sl.pkl" NUM_EVAL_STEP=10 \
  LOAD_INITIAL_MODEL=True INITIAL_MODEL_PATH="bridge_models/model-sl.pkl" \
  LOG_PATH="rl_log" EXP_NAME=pretrained-rl SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```
## `model-pretrained-rl-with-fsp.pkl`
Initialized with the WBridge5 supervised learning model, 
this approach trains using PPO with a self-play method called Fictitious Self Play (FSP), 
where opponents are randomly sampled from its own past models.  
Training settings can be configured with the following command:
```bash
python ppo.py NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
  TOTAL_TIMESTEPS=2621440000 UPDATE_EPOCHS=10 LR=0.000001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
  VE_COEF=0.5 NUM_EVAL_ENVS=100 EVAL_OPP_MODEL_PATH="bridge_models/model-sl.pkl" NUM_EVAL_STEP=10 \
  LOAD_INITIAL_MODEL=True INITIAL_MODEL_PATH="bridge_models/model-sl.pkl" \
  LOG_PATH="rl_log" EXP_NAME=pretrained-rl-with-fsp ratio_model_zoo=1 SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```
## `model-pretrained-rl-with-pfsp.pkl`
Initialized with the WBridge5 supervised learning model, 
this method employs a hybrid self-play approach for training with PPO. 
It alternates randomly between a simple self-play method and Prioritized Fictitious Self Play (PFSP), 
where opponents that the model struggles against are sampled preferentially from its own past models.  
Training settings can be configured with the following command:
```bash
python ppo.py NUM_ENVS=8192 NUM_STEPS=32 MINIBATCHE_SIZE=1024 \
  TOTAL_TIMESTEPS=2621440000 UPDATE_EPOCHS=10 LR=0.000001 GAMMA=1 GAE_LAMBDA=0.95 ENT_COEF=0.001 \
  VE_COEF=0.5 NUM_EVAL_ENVS=100 EVAL_OPP_MODEL_PATH="bridge_models/model-sl.pkl" NUM_EVAL_STEP=10 \
  LOAD_INITIAL_MODEL=True INITIAL_MODEL_PATH="bridge_models/model-sl.pkl" \
  LOG_PATH="rl_log" EXP_NAME=pretrained-rl-with-fsp ratio_model_zoo=0.3 prioritized_fictitious=True \
  SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```
