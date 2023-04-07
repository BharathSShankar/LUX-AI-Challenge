from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
import optax
from tqdm import tqdm

from agentHelpers.agent_controller import OverallController
from agentHelpers.agent_wrappers import JuxWrapperEnv

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from rlax import truncated_generalized_advantage_estimation

from neuralNets.actorNet import ActorNet
from neuralNets.criticNet import CriticNet
from neuralNets.PPOMemory import PPOMemory
from agentHelpers.agent_early_bidding import bidding, fact_placement_score

class nnAgentTrainer:

    def __init__(self,
                 player: str,
                 env_cfg: EnvConfig,
                 controller: OverallController,
                 actor: ActorNet,
                 critic: CriticNet,
                 memory: PPOMemory,
                 actor_params, critic_params,
                 actor_opt, critic_opt,
                 actor_state, critic_state
                 ) -> None:

        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.rng = jax.random.PRNGKey(0)
        self.env_cfg: EnvConfig = env_cfg
        self.controller = controller
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.critic_opt = critic_opt
        self.actor_opt = actor_opt
        self.actor_state = actor_state
        self.critic_state = critic_state
        self.loss_val_and_grad = jax.value_and_grad(
            self.loss_fn_ppo, argnums=(3, 4))

    def bid_policy(self, step: int, gameState, remainingOverageTime: int = 60):
        state_rep = JuxWrapperEnv.convert_obs(gameState, self.player)
        return bidding(state_rep["IMG"])

    def factory_placement_policy(self, step: int, gameState, remainingOverageTime: int = 60):
        if my_turn_to_place_factory(self.player != "player_0", step):
            state_rep = JuxWrapperEnv.convert_obs(gameState, self.player)
            mapScores = fact_placement_score(state_rep["IMG"])
            mapScores = mapScores.at[~gameState.board.valid_spawns_mask].set(-10000000000000000.0)
            mapScores = mapScores.reshape(-1)
            sample = jax.random.categorical(self.rng, mapScores).item()
            self.rng, _ = jax.random.split(self.rng)
            return {"spawn" : (sample // 48, sample % 48), "water" : 150 - 20 // self.env_cfg.MAX_FACTORIES, "metal":150 - 20 // self.env_cfg.MAX_FACTORIES} 
        return {}

    def choose_act(self, step: int, gameState, remainingOverageTime: int = 60):

        obs_proc = JuxWrapperEnv.convert_obs(gameState, self.player)

        to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, unit_actions_disc_params_N = self.actor.apply(
            self.actor_params,
            obs_proc["GIV"],
            obs_proc["IMG"],
            obs_proc["FACT"],
            obs_proc["UNIT"]
        )

        actions, act_probs = self.controller.convert_output_to_actions(
            self.player, gameState, to_change, unit_actions_logits, 
            fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, 
            unit_actions_disc_params_N, obs_proc["ACT_FACTS"], obs_proc["ACT_UNITS"]
        )

        value = self.critic.apply(
            self.critic_params,
            obs_proc["GIV"],
            obs_proc["IMG"]
        ).item()

        return actions, act_probs, value, (self.player, gameState, to_change, unit_actions_logits, 
            fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, 
            unit_actions_disc_params_N, obs_proc["GIV"], obs_proc["IMG"],
            obs_proc["FACT"], obs_proc["UNIT"], obs_proc["ACT_FACTS"], obs_proc["ACT_UNITS"])

    def act(self, step: int, gameState, remainingOverageTime: int = 60):

        obs_proc = JuxWrapperEnv.convert_obs(gameState, self.player)

        to_change, unit_actions_logits, fact_actions_logits, unit_actions_disc_params_R, unit_actions_disc_params_Rep, unit_actions_disc_params_N = self.actor.apply(
            self.actor_params,
            obs_proc["GIV"],
            obs_proc["IMG"],
            obs_proc["FACT"],
            obs_proc["UNIT"]
        )

        actions, _ = self.controller.convert_output_to_actions(
            self.player, gameState, to_change, unit_actions_logits,
            fact_actions_logits, unit_actions_disc_params_R,
            unit_actions_disc_params_Rep, unit_actions_disc_params_N,
            obs_proc["ACT_FACTS"], obs_proc["ACT_UNITS"]
        )

        return actions

    @partial(jax.jit, static_argnums=(0,2))
    def update_step(self, grad, opt, state, params):
        updates, new_state = opt.update(grad, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_state, new_params

    def learn(self, n_epochs, gamma, gae_lambda, eps):
        for _ in range(n_epochs):
            loss, (grad_loss_act, grad_loss_crit) = self.loss_val_and_grad(
                gamma, gae_lambda, eps, self.actor_params, self.critic_params)
            self.critic_state, self.critic_params = self.update_step(
                grad_loss_crit, self.critic_opt, self.critic_state, self.critic_params)
            self.actor_state, self.actor_params = self.update_step(
                grad_loss_act, self.actor_opt, self.actor_state, self.actor_params)
        self.memory.clear_memory()
        return self.actor_params, self.actor_state, self.critic_params, self.critic_state
        
    def loss_fn_ppo(self, gamma, gae_lambda, eps, actor_params, critic_params):
        to_change_old, global_states, img_states, fact_states, unit_states, fact_actions,\
            unit_actions, fact_probs, unit_probs, unit_actions_R, unit_actions_N, unit_actions_Rep, vals, rewards,\
            dones, avail_facts, avail_units, batches = self.memory.generate_batches()

        disc_factor_t = gamma ** jnp.arange(len(rewards))
        gae_val = truncated_generalized_advantage_estimation(
            rewards[1:],
            disc_factor_t[1:],
            gae_lambda,
            vals
        )
        loss = 0
        for batch in batches:
            loss += self.process_batch(to_change_old, global_states, img_states,
                                       fact_states, unit_states, fact_probs, unit_probs,
                                       unit_actions_R, unit_actions_N, unit_actions_Rep, avail_facts,
                                       avail_units, batch, gae_val, vals, eps, actor_params, critic_params)
        return loss
    @partial(jax.jit, static_argnums = (0,))
    def process_batch(self, to_change_old, global_states, img_states,
                      fact_states, unit_states, fact_probs, unit_probs,
                      unit_actions_R, unit_actions_N, unit_actions_Rep, avail_facts,
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
            unit_actions_disc_params_N = jax.vmap(self.actor.apply, in_axes = [None, 0, 0, 0 ,0])(
                actor_params, batch_global_states, batch_img_states, batch_fact_states, batch_unit_states)

        new_values = jax.vmap(self.critic.apply, in_axes=[None, 0, 0])(
            critic_params,  batch_global_states, batch_img_states)
        total_actor_loss = 0

        fact_loss = jnp.exp(jax.nn.log_softmax(fact_actions_logits, axis=2) -
                     jax.nn.log_softmax(batch_fact_probs, axis=2)).reshape((-1, 40, 3))* batch_fact_exist[:, :, jnp.newaxis]
        weighted_probs_fact = fact_loss * gae_val[batch, jnp.newaxis, jnp.newaxis]
        weighted_probs_fact_clip = jnp.clip(
            fact_loss, 1 - eps, 1 + eps) * gae_val[batch, jnp.newaxis, jnp.newaxis]
        total_actor_loss += - \
            jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(
                axis=(0, 2)).sum()

        unit_loss = jnp.exp(jax.nn.log_softmax(unit_actions_logits, axis=2) -
                     jax.nn.log_softmax(batch_unit_probs, axis=2)).reshape((-1, 300,20, 13)) * batch_unit_exist[:, :, jnp.newaxis, jnp.newaxis]
        weighted_probs_fact = unit_loss * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        weighted_probs_fact_clip = jnp.clip(
            unit_loss, 1 - eps, 1 + eps) * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        total_actor_loss += - \
            jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(
                axis=(0, 2)).sum()

        unit_loss = jnp.exp(jax.nn.log_softmax(unit_actions_disc_params_R,
                     axis=2) - jax.nn.log_softmax(batch_unit_probs_R, axis=2))
        weighted_probs_fact = unit_loss * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        weighted_probs_fact_clip = jnp.clip(
            unit_loss, 1 - eps, 1 + eps) * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        total_actor_loss += - \
            jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(
                axis=(0, 2)).sum()

        unit_loss = jnp.exp(jax.nn.log_softmax(unit_actions_disc_params_Rep,
                     axis=2) - jax.nn.log_softmax(batch_unit_probs_Rep, axis=2))
        weighted_probs_fact = unit_loss * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        weighted_probs_fact_clip = jnp.clip(
            unit_loss, 1 - eps, 1 + eps) * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        total_actor_loss += - \
            jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(
                axis=(0, 2)).sum()

        unit_loss = jnp.exp(jax.nn.log_softmax(unit_actions_disc_params_N,
                     axis=2) - jax.nn.log_softmax(batch_unit_probs_N, axis=2))
        weighted_probs_fact = unit_loss * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        weighted_probs_fact_clip = jnp.clip(
            unit_loss, 1 - eps, 1 + eps) * gae_val[batch, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        total_actor_loss += - \
            jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(
                axis=(0, 2)).sum()

        unit_loss = jnp.exp(jax.nn.log_softmax(to_change_new, axis=2) -
                     jax.nn.log_softmax(batch_to_change, axis=2))
        weighted_probs_fact = unit_loss * gae_val[batch, jnp.newaxis]
        weighted_probs_fact_clip = jnp.clip(
            unit_loss, 1 - eps, 1 + eps) * gae_val[batch, jnp.newaxis]
        total_actor_loss += - \
            jnp.minimum(weighted_probs_fact, weighted_probs_fact_clip).mean(
                axis=(0, 2)).sum()

        returns = gae_val[batch] + vals[batch, jnp.newaxis]
        critic_loss = jnp.mean((returns - new_values)**2)
        print(total_actor_loss.shape)
        total_loss = total_actor_loss + 0.5 * critic_loss
        return total_loss
