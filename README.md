# brl
reinforcement learning for bridge

## Usage
### Supervised Learning from Wbridge5 datasets
Please download the files "train.txt" and "test.txt" from the following URL and place them in the `your_data_directory`.
https://console.cloud.google.com/storage/browser/openspiel-data/bridge  

Run supervised learning
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

Examples
Run reinforcement learning without load initial model

```bash
python new_ppo.py EXP_NAME= EVAL_OPP_MODEL_PATH=$EVAL_OPP_MODEL_PATH UPDATE_EPOCHS=$UPDATE_EPOCHS LR=$LR LOAD_INITIAL_MODEL=$LOAD_INITIAL_MODEL TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS NUM_UPDATES=$NUM_UPDATES 
```

```
LR: float = 0.000001
NUM_ENVS: int = 8192
NUM_STEPS: int = 32
TOTAL_TIMESTEPS: int = 2_621_440_000
UPDATE_EPOCHS: int = 10  
NUM_MINIBATCHES: int = 128
NUM_UPDATES: int = 10000  
MINIBATCH_SIZE: int = 1024
# dataset config
DDS_RESULTS_DIR: str = "dds_results"
HASH_SIZE: int = 100_000
TRAIN_SIZE: int = 12_500_000
HASH_TABLE_NUM: int = 125
# eval config
NUM_EVAL_ENVS: int = 10000
EVAL_OPP_ACTIVATION: str = "relu"
EVAL_OPP_MODEL_TYPE: Literal["DeepMind", "FAIR"] = "DeepMind"
EVAL_OPP_MODEL_PATH: str = None
NUM_EVAL_STEP: int = 10
# log config
SAVE_MODEL: bool = False
SAVE_MODEL_INTERVAL: int = 1
LOG_PATH: str = "rl_log"
EXP_NAME: str = "exp_0000"
MODEL_SAVE_PATH: str = "rl_params"
TRACK: bool = True

# actor config
LOAD_INITIAL_MODEL: bool = False
INITIAL_MODEL_PATH: str = None
ACTOR_ACTIVATION: str = "relu"
ACTOR_MODEL_TYPE: Literal["DeepMind", "FAIR"] = "DeepMind"
# opposite config
GAME_MODE: Literal["competitive", "free-run"] = "competitive"
SELF_PLAY: bool = True
OPP_ACTIVATION: str = "relu"
OPP_MODEL_TYPE: Literal["DeepMind", "FAIR"] = "DeepMind"
OPP_MODEL_PATH: str = None
MODEL_ZOO_RATIO: float = 0
MODEL_ZOO_NUM: int = 50_000
MODEL_ZOO_THRESHOLD: float = -24
PRIORITIZED_FICTITIOUS: bool = False
PRIOR_T: float = 1
NUM_PRIORITIZED_ENVS: int = 100

# GAE config
GAMMA: float = 1
GAE_LAMBDA: float = 0.95
# loss config
CLIP_EPS: float = 0.2
ENT_COEF: float = 0.001
VF_COEF: float = 0.5
# PPO code optimaization
VALUE_CLIPPING: bool = True
GLOBAL_GRADIENT_CLIPPING: bool = True
ANNEAL_LR: bool = False  # True
REWARD_SCALING: bool = False
MAX_GRAD_NORM: float = 0.5
REWARD_SCALE: float = 7600

# illegal action config
ACTOR_ILLEGAL_ACTION_MASK: bool = True
ACTOR_ILLEGAL_ACTION_PENALTY: bool = False
ILLEGAL_ACTION_PENALTY: float = -1
ILLEGAL_ACTION_L2NORM_COEF: float = 0

NUM_EVAL_ENVS

TRACK
```
