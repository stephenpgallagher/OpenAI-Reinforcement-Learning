import os
import gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from rl2021.exercise3.agents import Agent
from rl2021.exercise3.networks import FCNetwork, Tanh2
from rl2021.exercise3.replay import Transition

def update_target(weights, target_weights, tau):
    '''
    Function to update target parameters slowly,
    based on the rate tau which is much less than 1
    '''
    # Obtaining parameter and parameter names
    model_params = weights.named_parameters()
    target_params = dict(target_weights.named_parameters())
    
    # Looping through model parameters
    for a, b in model_params:
        # Checking for model parameters in target parameters
        if a in target_params:
            # Updating the target parameters
            target_params[a].data.copy_(b.data * tau + (1-tau) * target_params[a].data)
    # Load updated parameters
    target_weights.load_state_dict(target_params)

class DDPG(Agent):
    """ DDPG

        ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

        :attr critic (FCNetwork): fully connected critic network
        :attr critic_optim (torch.optim): PyTorch optimiser for critic network
        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for actor network
        :attr gamma (float): discount rate gamma
        """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        ###  Building network and optimisers
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=Tanh2
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=Tanh2
        )

        self.actor_target.hard_update(self.actor)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)
        self.loss = torch.nn.MSELoss(reduction='sum')# Add in loss function using mean squared error
        ###


        ### Write any extra hyperparamterers needed here
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau
        ###

        ### Define a gaussian that will be used for exploration
        self.gaussian = Normal(loc=0, scale=0.1)
        
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q4")
        ###

        ### Write any agent parameters here
        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )
        ###

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path


    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q4")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        obs = torch.FloatTensor(obs)
        # Select greedy action when explore is false
        moment = self.actor(obs)
        moment = moment + self.gaussian.sample(moment.shape) if explore else moment
        # Clip the action in the range of [−2, 2]
        moment = moment.clamp(-2,2)
        # Return the action the agent should perform
        return moment.detach().numpy()
    
        # raise NotImplementedError("Needed for Q4")

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN
        
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your networks and return the q_loss and the policy_loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q4")
        
        # Deactivate autograd engine
        with torch.no_grad():
            # Execute action a_t with batch of transitions
            a_t = self.actor_target(batch.next_states).clamp(-2,2)
        
        # Disable gradient calculation
        with torch.no_grad():
            # Computing the targets
            Q_t = self.critic_target(torch.cat([a_t, batch.next_states], axis=1))
        exp_return = batch.rewards + self.gamma*(1-batch.done)*Q_t
        
        # Calculating action value predicted by the critic network
        Q_pred = self.critic(torch.cat([batch.actions, batch.states], axis=1))
        
        ### Q-LOSS FUNCTION
        # Set gradients to zero (i.e. clears old gradients from the last step)
        self.critic.zero_grad()
        # Derivative of loss using backpropragation
        q_loss = self.loss(Q_pred, exp_return).backward()
        # Step based on parameter gradients
        self.critic_optim.step()
        
        # State action pairs
        s_a_pair = torch.cat([self.actor(batch.states).clamp(-2,2), batch.states], axis=1)
        
        # ACTOR LOSS from mean value in critic network
        # Set gradients to zero
        self.actor.zero_grad()
        # Derivative of loss using backpropragation
        p_loss = torch.sum(-self.critic(s_a_pair)).backward()
        # Step based on parameter gradients
        self.policy_optim.step()
        
        # Update critic & actor parameters
        update_target(self.critic, self.critic_target, self.tau)
        update_target(self.actor, self.actor_target, self.tau)
        
        return {"q_loss": q_loss,
                "p_loss": p_loss}