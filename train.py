from argparse import ArgumentParser
from datetime import datetime
import os
import random
import uuid
from gym import make
from gym.spaces import Box, Discrete
# import roboschool
from yaml import load
import yaml
import os
from models import build_diag_gauss_policy, build_mlp, build_multinomial_policy
from simulators import *
from transforms import *
from torch_utils import get_device
from trpo import TRPO as TRPOBase
from trpo_v1 import TRPO as TRPOV1


# Set the seed for all relevant components
seed = 42

# Set the seed for Python's random module
random.seed(seed)

# Set the seed for NumPy
np.random.seed(seed)

# Set the seed for PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optional: Set the seed for other libraries
os.environ['PYTHONHASHSEED'] = str(seed)
config_filename = 'config.yaml'

parser = ArgumentParser(prog='train.py',
                        description='Train a policy on the specified environment' \
                        ' using Trust Region Policy Optimization (Schulman 2015)' \
                        ' with Generalized Advantage Estimation (Schulman 2016).')
parser.add_argument('--continue', dest='continue_from_file', action='store_true',
                    help='Set this flag to continue training from a previously ' \
                    'saved session. Session will be overwritten if this flag is ' \
                    'not set and a saved file associated with model-name already ' \
                    'exists.')
parser.add_argument('--ver', type=str, required=True)
parser.add_argument('--model-name', type=str, dest='model_name', required=True,
                    help='The entry in config.yaml from which settings' \
                    'should be loaded.')

parser.add_argument('--high_t', type=float)
parser.add_argument('--damp_f', type=float)
parser.add_argument('--low_t', type=float)
parser.add_argument('--simulator', dest='simulator_type', type=str, default='single-path',
                    choices=['single-path', 'vine'], help='The type of simulator' \
                    ' to use when collecting training experiences.')

args = parser.parse_args()
version = args.ver
damp_factor = args.damp_f
continue_from_file = args.continue_from_file
model_name = args.model_name
simulator_type = args.simulator_type

all_configs = load(open(config_filename, 'r'), yaml.FullLoader)
config = all_configs[model_name]

device = get_device()
if args.low_t and args.high_t:
    low_t = torch.tensor(args.low_t).to(device)
    high_t = torch.tensor(args.high_t).to(device)


# Find the input size, hidden dim sizes, and output size
env_name = config['env_name']
env = make(env_name)
action_space = env.action_space
observation_space = env.observation_space
policy_hidden_dims = config['policy_hidden_dims']
vf_hidden_dims = config['vf_hidden_dims']
vf_args = (observation_space.shape[0] + 1, vf_hidden_dims, 1)

# Initialize the policy
if type(action_space) is Box:
    policy_args = (observation_space.shape[0], policy_hidden_dims, action_space.shape[0])
    policy = build_diag_gauss_policy(*policy_args)
elif type(action_space) is Discrete:
    policy_args = (observation_space.shape[0], policy_hidden_dims, action_space.n)
    policy = build_multinomial_policy(*policy_args)
else:
    raise NotImplementedError

# Initalize the value function
value_fun = build_mlp(*vf_args)
policy.to(device)
value_fun.to(device)
# Initialize the state transformation
z_filter = ZFilter()
state_bound = Bound(-5, 5)
state_filter = Transform(state_bound, z_filter)

# Initialize the simulator
n_trajectories = config['n_trajectories']
max_timesteps = config['max_timesteps']
try:
    env_args = config['env_args']
except:
    env_args = {}

if simulator_type == 'single-path':
    simulator = SinglePathSimulator(env_name, policy, n_trajectories,
                                    max_timesteps, state_filter=state_filter,
                                    **env_args)
elif simulator_type == 'vine':
    raise NotImplementedError
try:
    trpo_args = config['trpo_args']
except:
    trpo_args = {}


# Get the current timestamp (current time with timezone)
timestamp = datetime.now()

# If you need the timestamp in a specific format:
timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
print(f'Version is : {version}')
if version == 'base':
    exp_dir = os.path.join('experiments', f'{model_name}_{timestamp}_seed_{seed}_ver_{version}')
    os.makedirs(exp_dir)
    trpo = TRPOBase(policy, value_fun, simulator, model_name=model_name,
                    continue_from_file=continue_from_file, experiment_dir=exp_dir, **trpo_args, save_every=50)
elif version == 'filtering':
    exp_dir = os.path.join('experiments', f'{model_name}_{timestamp}_seed_{seed}_ver_{version}_high_t_{high_t:.2f}_low_t_{low_t:.2f}_damp_f_{damp_factor:.2f}')
    os.makedirs(exp_dir)
    trpo = TRPOV1(policy, value_fun, simulator, model_name=model_name,
                  continue_from_file=continue_from_file, experiment_dir=exp_dir, **trpo_args, save_every=50 ,low_t=low_t, high_t=high_t, damp_factor = damp_factor) 
else:
    raise NotImplementedError()


print(f'Training policy {model_name} on {env_name} environment...\n')



trpo.train(config['n_episodes'])
trpo.writer.close()
print('\nTraining complete.\n')
