from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from agentHelpers.agent_controller import OverallController
from agentHelpers.agent_obs_processing import map_2_vec
from agentHelpers.agent_wrappers import JuxWrapperEnv

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from rlax import truncated_generalized_advantage_estimation

from neuralNets.actorNet import ActorNet
from neuralNets.criticNet import CriticNet
from neuralNets.PPOMemory import PPOMemory
from neuralNets.mapScoring import MapScorer

class nnAgentTrainer(core.Actor):

    def __init__(self, 
                player: str,
                env_cfg: EnvConfig, 
                controller: OverallController, 
                actor: ActorNet, 
                critic: CriticNet, 
                memory: PPOMemory,
                mapScorer: MapScorer,
                actor_params, critic_params, mapScorer_params
        ) -> None:

        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.controller = controller
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.mapScorer = mapScorer
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.mapScorer_params = mapScorer_params

        self.loss_val_and_grad = jax.value_and_grad(self.loss_fn_ppo, argnums=(4, 5))

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        if my_turn_to_place_factory(self.player == "player_0", step): 
            gameState = obs_to_game_state(step, self.env_cfg, obs)
            map_vec = map_2_vec(gameState)
            map_val = self.factory_policy.apply(self.factory_weights, map_vec)
            map_val[~gameState.board.valid_spawns_mask] = -np.inf
            idx = np.argmax(map_val)
            pos = idx // gameState.env_cfg.map_size, idx % gameState.env_cfg.map_size
            return dict(spawn = pos, metal = 150, water = 150)
        return {}


    def choose_act(self, step: int, obs, remainingOverageTime: int = 60):

        gameState = obs_to_game_state(step, self.env_cfg, obs)
        obs_proc = JuxWrapperEnv.convert_obs(gameState, self.player)

        to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, unit_actions_disc_params_N = self.actor.apply(
            self.actor_weights, 
            obs_proc["GIV"],
            obs_proc["IMG"],
            obs_proc["FACT"],
            obs_proc["UNIT"]    
        )
        
        actions, act_probs = self.controller.convert_output_to_actions(
            self.player, gameState, to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, unit_actions_disc_params_N, obs_proc["ACT_FACTS"], obs_proc["ACT_UNITS"]
        )

        value = self.critic.apply(
            self.critic_params,
            obs_proc["GIV"],
            obs_proc["IMG"]
        )

        return actions, act_probs, value
    
    def act(self, step: int, obs, remainingOverageTime: int = 60):

        gameState = obs_to_game_state(step, self.env_cfg, obs)
        obs_proc = JuxWrapperEnv.convert_obs(gameState, self.player)

        to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, unit_actions_disc_params_N = self.actor.apply(
            self.actor_weights, 
            obs_proc["GIV"],
            obs_proc["IMG"],
            obs_proc["FACT"],
            obs_proc["UNIT"]    
        )
        
        actions, _ = self.controller.convert_output_to_actions(
            self.player, gameState, to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, unit_actions_disc_params_N
        ) 

        return actions
    
    def learn(self, n_epochs, gamma, gae_lambda, eps):
        for _ in tqdm(range(n_epochs)):
            loss, grad_loss = self.loss_val_and_grad(gamma, gae_lambda, eps, self.actor_params, self.critic_params)
            

    @partial(jax.jit, static_argnums = (0, 1, 2, 3,))
    def loss_fn_ppo(self, gamma, gae_lambda, eps, actor_params, critic_params):
        to_change_old, global_states, img_states, fact_states, unit_states, fact_actions,\
                unit_actions, fact_probs, unit_probs, unit_actions_R, unit_actions_N, unit_actions_Rep, vals, rewards,\
                dones, batches, avail_facts, avail_units = self.memory.generate_batches()
            
        disc_factor_t = gamma ** jnp.arange(len(rewards))

        gae_val = truncated_generalized_advantage_estimation(
                jnp.array(rewards),
                disc_factor_t,
                gae_lambda,
                jnp.array(vals)
            )
        loss = 0
        for batch in batches:
            loss += self.process_batch(to_change_old, global_states, img_states, \
                        fact_states, unit_states, fact_probs, unit_probs,\
                        unit_actions_R, unit_actions_N, unit_actions_Rep, avail_facts,\
                        avail_units, batch, gae_val, vals, eps)
        return loss
    

    def process_batch(self, to_change_old, global_states, img_states, \
                        fact_states, unit_states, fact_probs, unit_probs,\
                        unit_actions_R, unit_actions_N, unit_actions_Rep, avail_facts,\
                        avail_units, batch, gae_val, vals, eps, actor_params, critic_params):
        batch_to_change = to_change_old[batch]
        batch_global_states = global_states[batch]
        batch_img_states = img_states[batch]
        batch_fact_states = fact_states[batch]            
        batch_unit_states = unit_states[batch]

        batch_fact_probs = fact_probs[batch]
        batch_unit_probs = unit_probs[batch]
        batch_unit_probs_R = unit_actions_R[batch]
        batch_unit_probs_N = unit_actions_N[batch]
        batch_unit_probs_Rep = unit_actions_Rep[batch]

        batch_fact_exist = avail_facts[batch]
        batch_unit_exist = avail_units[batch]

        to_change_new, unit_actions_logits, fact_actions_logits, \
            unit_actions_disc_params_R, unit_actions_disc_params_Rep,\
                  unit_actions_disc_params_N = self.actor.apply(actor_params, batch_global_states, batch_img_states, batch_fact_states, batch_unit_states)
            
        new_values = self.critic.apply(critic_params,  batch_global_states, batch_img_states)
        total_actor_loss = 0

        fact_loss = (jax.nn.log_softmax(fact_actions_logits, axis = 2) - jax.nn.log_softmax(batch_fact_probs, axis = 2)).exp() * batch_fact_exist
        weighted_probs_fact = fact_loss * gae_val[batch]
        weighted_probs_fact_clip = jnp.clip(fact_loss, 1 - eps, 1 + eps) * gae_val[batch]
        total_actor_loss += - jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(axis = (0, 2)).sum()            

        unit_loss = (jax.nn.log_softmax(unit_actions_logits, axis=2) - jax.nn.log_softmax(batch_unit_probs, axis=2)).exp() * batch_unit_exist
        weighted_probs_fact = unit_loss * gae_val[batch]
        weighted_probs_fact_clip = jnp.clip(unit_loss, 1 - eps, 1 + eps) * gae_val[batch]
        total_actor_loss += -jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(axis=(0, 2)).sum()

        unit_loss = (jax.nn.log_softmax(unit_actions_disc_params_R, axis=2) - jax.nn.log_softmax(batch_unit_probs_R, axis=2)).exp()
        weighted_probs_fact = unit_loss * gae_val[batch]
        weighted_probs_fact_clip = jnp.clip(unit_loss, 1 - eps, 1 + eps) * gae_val[batch]
        total_actor_loss += -jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(axis=(0, 2)).sum()

        unit_loss = (jax.nn.log_softmax(unit_actions_disc_params_Rep, axis=2) - jax.nn.log_softmax(batch_unit_probs_Rep, axis=2)).exp()
        weighted_probs_fact = unit_loss * gae_val[batch]
        weighted_probs_fact_clip = jnp.clip(unit_loss, 1 - eps, 1 + eps) * gae_val[batch]
        total_actor_loss += -jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(axis=(0, 1, 2)).sum()

        unit_loss = (jax.nn.log_softmax(unit_actions_disc_params_N, axis=2) - jax.nn.log_softmax(batch_unit_probs_N, axis=2)).exp()
        weighted_probs_fact = unit_loss * gae_val[batch]
        weighted_probs_fact_clip = jnp.clip(unit_loss, 1 - eps, 1 + eps) * gae_val[batch]
        total_actor_loss += -jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(axis=(0, 1, 2)).sum()  

        unit_loss = (jax.nn.log_softmax(to_change_new, axis=2) - jax.nn.log_softmax(batch_to_change, axis=2)).exp()
        weighted_probs_fact = unit_loss * gae_val[batch]
        weighted_probs_fact_clip = jnp.clip(unit_loss, 1 - eps, 1 + eps) * gae_val[batch]
        total_actor_loss += -jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(axis=(0, 1, 2)).sum()    

        returns = gae_val[batch] + vals[batch]
        critic_loss = jnp.mean((returns - new_values)**2)

        total_loss = total_actor_loss + 0.5 * critic_loss
        return total_loss