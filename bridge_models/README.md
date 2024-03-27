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
python ppo.py num_envs=8192 num_steps=32 minibatch_size=1024 \
  total_timesteps=5242880000 update_epochs=10 lr=0.00001 gamma=1 gae_lambda=0.95 ent_coef=0.001 \
  VE_COEF=0.5 num_eval_envs=100 eval_opp_model_path="bridge_models/model-sl.pkl" num_eval_step=10 \
  load_initial_model=False log_path="rl_log" exp_name=from-scratch-rl save_model=True save_model_interval=100
```
## `model-pretrained-rl.pkl`
Initialized with the WBridge5 supervised learning model and trained using PPO with a simple self-play method.  
Training settings can be configured with the following command:
```bash
python ppo.py num_envs=8192 num_steps=32 minibatch_size=1024 \
  total_timesteps=2621440000 update_epochs=10 lr=0.000001 gamma=1 gae_lambda=0.95 ent_coef=0.001 \
  VE_COEF=0.5 num_eval_envs=100 eval_opp_model_path="bridge_models/model-sl.pkl" num_eval_step=10 \
  load_initial_model=True initial_model_path="bridge_models/model-sl.pkl" \
  log_path="rl_log" exp_name=pretrained-rl save_model=True save_model_interval=100
```
## `model-pretrained-rl-with-fsp.pkl`
Initialized with the WBridge5 supervised learning model, 
this approach trains using PPO with a self-play method called Fictitious Self Play (FSP), 
where opponents are randomly sampled from its own past models.  
Training settings can be configured with the following command:
```bash
python ppo.py num_envs=8192 num_steps=32 minibatch_size=1024 \
  total_timesteps=2621440000 update_epochs=10 lr=0.000001 gamma=1 gae_lambda=0.95 ent_coef=0.001 \
  VE_COEF=0.5 num_eval_envs=100 eval_opp_model_path="bridge_models/model-sl.pkl" num_eval_step=10 \
  load_initial_model=True initial_model_path="bridge_models/model-sl.pkl" \
  log_path="rl_log" exp_name=pretrained-rl-with-fsp ratio_model_zoo=1 save_model=True save_model_interval=100
```
## `model-pretrained-rl-with-pfsp.pkl`
Initialized with the WBridge5 supervised learning model, 
this method employs a hybrid self-play approach for training with PPO. 
It alternates randomly between a simple self-play method and Prioritized Fictitious Self Play (PFSP), 
where opponents that the model struggles against are sampled preferentially from its own past models.  
Training settings can be configured with the following command:
```bash
python ppo.py num_envs=8192 num_steps=32 minibatch_size=1024 \
  total_timesteps=2621440000 update_epochs=10 lr=0.000001 gamma=1 gae_lambda=0.95 ent_coef=0.001 \
  VE_COEF=0.5 num_eval_envs=100 eval_opp_model_path="bridge_models/model-sl.pkl" num_eval_step=10 \
  load_initial_model=True initial_model_path="bridge_models/model-sl.pkl" \
  log_path="rl_log" exp_name=pretrained-rl-with-pfsp ratio_model_zoo=0.3 prioritized_fictitious=True \
  SAVE_MODEL=True SAVE_MODEL_INTERVAL=100
```
