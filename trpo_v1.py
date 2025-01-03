from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
import yaml
import torch
from torch.nn import MSELoss
from torch.optim import LBFGS
from yaml import load
from conjugate_gradient import cg_solver
from distribution_utils import mean_kl_first_fixed
from hvp import get_Hvp_fun
from torch.utils.tensorboard import SummaryWriter
from torch_utils import apply_update, flat_grad, get_device, get_flat_params

config = load(open('config.yaml', 'r'), yaml.FullLoader)
# TO DO: Model name issue

class TRPO:
    '''
    Optimizes the given policy using Trust Region Policy Optization (Schulman 2015)
    with Generalized Advantage Estimation (Schulman 2016).

    Attributes
    ----------
    policy : torch.nn.Sequential
        the policy to be optimized

    value_fun : torch.nn.Sequential
        the value function to be optimized and used when calculating the advantages

    simulator : Simulator
        the simulator to be used when generating training experiences

    max_kl_div : float
        the maximum kl divergence of the policy before and after each step

    max_value_step : float
        the learning rate for the value function

    vf_iters : int
        the number of times to optimize the value function over each set of
        training experiences

    vf_l2_reg_coef : float
        the regularization term when calculating the L2 loss of the value function

    discount : float
        the coefficient to use when discounting the rewards

    lam : float
        the bias reduction parameter to use when calculating advantages using GAE

    cg_damping : float
        the multiple of the identity matrix to add to the Hessian when calculating
        Hessian-vector products

    cg_max_iters : int
        the maximum number of iterations to use when solving for the optimal
        search direction using the conjugate gradient method

    line_search_coef : float
        the proportion by which to reduce the step length on each iteration of
        the line search

    line_search_max_iters : int
        the maximum number of line search iterations before returning 0.0 as the
        step length

    line_search_accept_ratio : float
        the minimum proportion of error to accept from linear extrapolation when
        doing the line search

    mse_loss : torch.nn.MSELoss
        a MSELoss object used to calculating the value function loss

    value_optimizer : torch.optim.LBFGS
        a LBFGS object used to optimize the value function

    model_name : str
        an identifier for the model to be used when generating filepath names

    continue_from_file : bool
        whether to continue training from a previous saved session

    save_every : int
        the number of training iterations to go between saving the training session

    episode_num : int
        the number of episodes already completed

    elapsed_time : datetime.timedelta
        the elapsed training time so far

    device : torch.device
        device to be used for pytorch tensor operations

    mean_rewards : list
        a list of the mean rewards obtained by the agent for each episode so far

    Methods
    -------
    train(n_episodes)
        train the policy and value function for the n_episodes episodes

    unroll_samples(samples)
        unroll the samples generated by the simulator and return a flattend
        version of all states, actions, rewards, and estimated Q-values

    get_advantages(samples)
        return the GAE advantages and a version of the unrolled states with
        a time variable concatenated to each state

    update_value_fun(states, q_vals)
        calculate one update step and apply it to the value function

    update_policy(states, actions, advantages)
        calculate one update step using TRPO and apply it to the policy

    surrogate_loss(log_action_probs, imp_sample_probs, advantages)
        calculate the loss for the policy on a batch of experiences

    get_max_step_len(search_dir, Hvp_fun, max_step, retain_graph=False)
        calculate the coefficient for search_dir s.t. the change in the function
        approximator of interest will be equal to max_step

    save_session()
        save the current training session

    load_session()
        load a previously saved training session

    print_update()
        print an update message that displays statistics about the most recent
        training iteration
    '''
    def __init__(self, policy, value_fun, simulator, experiment_dir, max_kl_div=0.01, max_value_step=0.01,
                vf_iters=1, vf_l2_reg_coef=1e-3, discount=0.995, lam=0.98, cg_damping=1e-3,
                cg_max_iters=10, line_search_coef=0.9, line_search_max_iter=10,
                line_search_accept_ratio=0.1, model_name=None, continue_from_file=False,
                save_every=20, low_t=-0.7, high_t=0.4, damp_factor= 0.9):
        '''
        Parameters
        ----------

        policy : torch.nn.Sequential
            the policy to be optimized

        value_fun : torch.nn.Sequential
            the value function to be optimized and used when calculating the advantages

        simulator : Simulator
            the simulator to be used when generating training experiences

        max_kl_div : float
            the maximum kl divergence of the policy before and after each step
            (default is 0.01)

        max_value_step : float
            the learning rate for the value function (default is 0.01)

        vf_iters : int
            the number of times to optimize the value function over each set of
            training experiences (default is 1)

        vf_l2_reg_coef : float
            the regularization term when calculating the L2 loss of the value function
            (default is 0.001)

        discount : float
            the coefficient to use when discounting the rewards (discount is 0.995)

        lam : float
            the bias reduction parameter to use when calculating advantages using GAE
            (default is 0.98)

        cg_damping : float
            the multiple of the identity matrix to add to the Hessian when calculating
            Hessian-vector products (default is 0.001)

        cg_max_iters : int
            the maximum number of iterations to use when solving for the optimal
            search direction using the conjugate gradient method (default is 10)

        line_search_coef : float
            the proportion by which to reduce the step length on each iteration of
            the line search (default is 0.9)

        line_search_max_iters : int
            the maximum number of line search iterations before returning 0.0 as the
            step length (default is 10)

        line_search_accept_ratio : float
            the minimum proportion of error to accept from linear extrapolation when
            doing the line search (default is 0.1)

        model_name : str
            an identifier for the model to be used when generating filepath names
            (default is None)

        continue_from_file : bool
            whether to continue training from a previous saved session (default is False)

        save_every : int
            the number of training iterations to go between saving the training session
            (default is 1)
        '''

        self.policy = policy
        self.value_fun = value_fun
        self.simulator = simulator
        self.max_kl_div = max_kl_div
        self.damp_factor = damp_factor
        self.max_value_step = max_value_step
        self.vf_iters = vf_iters
        self.vf_l2_reg_coef = vf_l2_reg_coef
        self.discount = discount
        self.low_t= low_t
        self.high_t = high_t
        self.lam = lam
        self.cg_damping = cg_damping
        self.cg_max_iters = cg_max_iters
        self.line_search_coef = line_search_coef
        self.line_search_max_iter = line_search_max_iter
        self.line_search_accept_ratio = line_search_accept_ratio
        self.mse_loss = MSELoss(reduction='mean')
        self.value_optimizer = LBFGS(self.value_fun.parameters(), lr=max_value_step, max_iter=25)
        self.model_name = model_name
        self.continue_from_file = continue_from_file
        self.save_every = save_every
        self.episode_num = 0
        self.elapsed_time = timedelta(0)
        self.device = get_device()
        self.mean_rewards = []
        log_dir = os.path.join(experiment_dir, 'logs')
        os.makedirs(log_dir)
        self.save_dir = os.path.join(experiment_dir, 'saved_sessions')
        os.makedirs(self.save_dir)
        # TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        if not model_name and continue_from_file:
            raise Exception('Argument continue_from_file to __init__ method of ' \
                            'TRPO case was set to True but model_name was not ' \
                            'specified.')

        if not model_name and save_every:
            raise Exception('Argument save_every to __init__ method of TRPO ' \
                            'was set to a value greater than 0 but model_name ' \
                            'was not specified.')

        if continue_from_file:
            self.load_session()

    def train(self, n_episodes):
        last_q = None
        last_states = None

        while self.episode_num < n_episodes:
            start_time = dt.now()
            self.episode_num += 1
            samples = self.simulator.sample_trajectories()
            states, actions, rewards, q_vals = self.unroll_samples(samples)

            advantages, states_with_time = self.get_advantages(samples)
            advantages -= torch.mean(advantages)
            advantages /= torch.std(advantages)

            self.update_policy(states, actions, advantages)

            if last_q is not None:
                self.update_value_fun(torch.cat([states_with_time, last_states]), torch.cat([q_vals, last_q]))
            else:
                self.update_value_fun(states_with_time, q_vals)

            last_q = q_vals
            last_states = states_with_time

            mean_reward = np.mean([np.sum(trajectory['rewards']) for trajectory in samples])
            mean_reward_np = mean_reward
            self.mean_rewards.append(mean_reward_np)
            self.elapsed_time += dt.now() - start_time
            self.print_update()
            # Log metrics to TensorBoard
            self.writer.add_scalar("Reward/MeanReward", mean_reward, self.episode_num)
            self.writer.add_scalar("Time/ElapsedTime", self.elapsed_time.total_seconds(), self.episode_num)

            if self.save_every and not self.episode_num % self.save_every:
                self.save_session()

    def unroll_samples(self, samples):
        q_vals = []

        for trajectory in samples:
            rewards = torch.tensor(trajectory['rewards'])
            reverse = torch.arange(rewards.size(0) - 1, -1, -1)
            discount_pows = torch.pow(self.discount, torch.arange(0, rewards.size(0)).float())
            discounted_rewards = rewards * discount_pows
            disc_reward_sums = torch.cumsum(discounted_rewards[reverse], dim=-1)[reverse]
            trajectory_q_vals = disc_reward_sums / discount_pows
            q_vals.append(trajectory_q_vals)

        states = torch.cat([torch.stack(trajectory['states']) for trajectory in samples])
        actions = torch.cat([torch.stack(trajectory['actions']) for trajectory in samples])
        rewards = torch.cat([torch.stack(trajectory['rewards']) for trajectory in samples])
        q_vals = torch.cat(q_vals)

        return states, actions, rewards, q_vals

    def get_advantages(self, samples):
        advantages = []
        states_with_time = []
        T = self.simulator.trajectory_len

        for trajectory in samples:
            time = torch.arange(0, len(trajectory['rewards'])).unsqueeze(1).float() / T
            states = torch.stack(trajectory['states'])
            states = torch.cat([states, time], dim=-1)
            states = states.to(self.device)
            states_with_time.append(states.cpu())
            rewards = torch.tensor(trajectory['rewards'])

            state_values = self.value_fun(states)
            state_values = state_values.view(-1)
            state_values = state_values.cpu()
            state_values_next = torch.cat([state_values[1:], torch.tensor([0.0])])

            td_residuals = rewards + self.discount * state_values_next - state_values
            reverse = torch.arange(rewards.size(0) - 1, -1, -1)
            discount_pows = torch.pow(self.discount * self.lam, torch.arange(0, rewards.size(0)).float())
            discounted_residuals = td_residuals * discount_pows
            disc_res_sums = torch.cumsum(discounted_residuals[reverse], dim=-1)[reverse]
            trajectory_advs = disc_res_sums / discount_pows
            advantages.append(trajectory_advs)

        advantages = torch.cat(advantages)

        states_with_time = torch.cat(states_with_time)

        return advantages, states_with_time

    def update_value_fun(self, states, q_vals):
        self.value_fun.train()

        states = states.to(self.device)
        q_vals = q_vals.to(self.device)

        for i in range(self.vf_iters):
            def mse():
                self.value_optimizer.zero_grad()
                state_values = self.value_fun(states).view(-1)

                loss = self.mse_loss(state_values, q_vals)

                flat_params = get_flat_params(self.value_fun)
                l2_loss = self.vf_l2_reg_coef * torch.sum(torch.pow(flat_params, 2))
                loss += l2_loss

                loss.backward()

                return loss

            self.value_optimizer.step(mse)

            # Log value function loss
            self.writer.add_scalar("ValueFunction/Loss", mse().item(), self.episode_num)

    def update_policy(self, states, actions, advantages):
        self.policy.train()

        states = states.to(self.device)
        actions = actions.to(self.device)
        advantages = advantages.to(self.device)

        action_dists = self.policy(states)
        log_action_probs = action_dists.log_prob(actions)

        loss,  _ = self.surrogate_loss(log_action_probs, log_action_probs.detach(), advantages)
        loss_grad = flat_grad(loss, self.policy.parameters(), retain_graph=True)

        mean_kl = mean_kl_first_fixed(action_dists, action_dists)
        # Log KL divergence

        Fvp_fun = get_Hvp_fun(mean_kl, self.policy.parameters())
        search_dir = cg_solver(Fvp_fun, loss_grad, self.cg_max_iters)

        expected_improvement = torch.matmul(loss_grad, search_dir)
        damped_nums = []
        def constraints_satisfied(step, beta):
            apply_update(self.policy, step)

            with torch.no_grad():
                new_action_dists = self.policy(states)
                new_log_action_probs = new_action_dists.log_prob(actions)

                new_loss, damped_samples_count = self.surrogate_loss(new_log_action_probs, log_action_probs, advantages)
                damped_nums.append(damped_samples_count)
                mean_kl = mean_kl_first_fixed(action_dists, new_action_dists)
                self.writer.add_scalar("Policy/MeanKL", mean_kl.item(), self.episode_num)

            actual_improvement = new_loss - loss
            improvement_ratio = actual_improvement / (expected_improvement * beta)

            apply_update(self.policy, -step)

            surrogate_cond = improvement_ratio >= self.line_search_accept_ratio and actual_improvement > 0.0
            kl_cond = mean_kl <= self.max_kl_div

            return surrogate_cond and kl_cond
    
        max_step_len = self.get_max_step_len(search_dir, Fvp_fun, self.max_kl_div, retain_graph=True)
        step_len = self.line_search(search_dir, max_step_len, constraints_satisfied)

        opt_step = step_len * search_dir
        
        apply_update(self.policy, opt_step)
        self.writer.add_scalar("Policy/MeanLearningRate", step_len, self.episode_num)
        self.writer.add_scalar("Mean samples damped", np.mean(damped_nums), self.episode_num)

    def surrogate_loss(self, log_action_probs, imp_sample_probs, advantages):
        """
        log_action_probs: Log action probabilities from the policy
        imp_sample_probs: Importance sampling probabilities for the samples
        advantages: Advantage values for the samples
        """
        # Compute the difference (log_action_probs - imp_sample_probs)
        diff = log_action_probs - imp_sample_probs
        
        # Initialize damping factor (e.g., 1 means no damping, < 1 means damping)
        damp_mat = torch.ones_like(diff)  # Default is no damping (factor = 1)

        # Apply damping if the condition is not satisfied
        mask = (diff < self.low_t) | (diff > self.high_t)
        non_zero_count = torch.count_nonzero(mask).item() / mask.numel()
        damp_mat[(diff < self.low_t) | (diff > self.high_t)] = torch.tensor(self.damp_factor, dtype=torch.float32, device=self.device)

        # Compute the surrogate loss as before
        surrogate_loss = torch.exp(diff) * advantages

        # Apply the damping factor to the surrogate loss
        weighted_surrogate_loss = surrogate_loss * damp_mat

        # Return the mean of the weighted surrogate loss
        return torch.mean(weighted_surrogate_loss), non_zero_count
    def get_max_step_len(self, search_dir, Hvp_fun, max_step, retain_graph=False):
        num = 2 * max_step
        denom = torch.matmul(search_dir, Hvp_fun(search_dir, retain_graph))
        max_step_len = torch.sqrt(num / denom)

        return max_step_len

    def save_session(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        save_path = os.path.join(self.save_dir, self.model_name + '.pt')

        ckpt = {'policy_state_dict': self.policy.state_dict(),
                'value_state_dict': self.value_fun.state_dict(),
                'mean_rewards': self.mean_rewards,
                'episode_num': self.episode_num,
                'elapsed_time': self.elapsed_time}

        if self.simulator.state_filter:
            ckpt['state_filter'] = self.simulator.state_filter

        torch.save(ckpt, save_path)

        # Log checkpoint
        self.writer.add_scalar("Checkpoint/Episode", self.episode_num, self.episode_num)
    
    def line_search(self, search_dir, max_step_len, constraints_satisfied, line_search_coef=0.9,
                    max_iter=10):
            '''
            Perform a backtracking line search that terminates when constraints_satisfied
            return True and return the calculated step length. Return 0.0 if no step
            length can be found for which constraints_satisfied returns True

            Parameters
            ----------
            search_dir : torch.FloatTensor
                the search direction along which the line search is done

            max_step_len : torch.FloatTensor
                the maximum step length to consider in the line search

            constraints_satisfied : callable
                a function that returns a boolean indicating whether the constraints
                are met by the current step length

            line_search_coef : float
                the proportion by which to reduce the step length after each iteration

            max_iter : int
                the maximum number of backtracks to do before return 0.0

            Returns
            -------
            the maximum step length coefficient for which constraints_satisfied evaluates
            to True
            '''
   
            step_len = max_step_len / line_search_coef

            for i in range(max_iter):
                step_len *= line_search_coef

                if constraints_satisfied(step_len * search_dir, step_len):
                    return step_len

            return torch.tensor(0.0).to(self.device)

    def load_session(self):
        load_path = os.path.join(self.save_dir, self.model_name + '.pt')
        ckpt = torch.load(load_path)

        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.value_fun.load_state_dict(ckpt['value_state_dict'])
        self.mean_rewards = ckpt['mean_rewards']
        self.episode_num = ckpt['episode_num']
        self.elapsed_time = ckpt['elapsed_time']

        try:
            self.simulator.state_filter = ckpt['state_filter']
        except KeyError:
            pass

    def print_update(self):
        update_message = '[EPISODE]: {0}\t[AVG. REWARD]: {1:.4f}\t [ELAPSED TIME]: {2}'
        elapsed_time_str = ''.join(str(self.elapsed_time).split('.')[0])
        format_args = (self.episode_num, self.mean_rewards[-1], elapsed_time_str)
        print(update_message.format(*format_args))
