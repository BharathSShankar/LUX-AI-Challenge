import numpy as np
import jax.numpy as jnp


class PPOMemory:
    def __init__(self, batch_size):
        self.to_change = []
        self.global_states = []
        self.img_states = []
        self.fact_states = []
        self.unit_states = []
        self.fact_actions = []
        self.unit_actions = []
        self.fact_probs = []
        self.unit_probs = []
        self.unit_probs_R = []
        self.unit_probs_N = []
        self.unit_probs_Rep = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.avail_facts = []
        self.avail_units = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.global_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (jnp.stack(self.to_change, axis=0),
                jnp.stack(self.global_states, axis=0),
                jnp.stack(self.img_states, axis=0),
                jnp.stack(self.fact_states, axis=0),
                jnp.stack(self.unit_states, axis=0),
                jnp.stack(self.fact_actions, axis=0),
                jnp.stack(self.unit_actions, axis=0),
                jnp.stack(self.fact_probs, axis=0),
                jnp.stack(self.unit_probs, axis=0),
                jnp.stack(self.unit_probs_R, axis=0),
                jnp.stack(self.unit_probs_N, axis=0),
                jnp.stack(self.unit_probs_Rep, axis=0),
                jnp.stack(self.vals, axis=0),
                jnp.stack(self.rewards, axis=0),
                jnp.stack(self.dones, axis=0),
                jnp.stack(self.avail_facts, axis=0),
                jnp.stack(self.avail_units, axis=0),
                batches)

    def store_memory(self, to_change, global_state, img_state, fact_state, unit_state, fact_action, unit_action, fact_prob, unit_prob, unit_prob_R, unit_prob_N, unit_prob_Rep, val, reward, done, avail_facts, avail_units):
        self.to_change.append(to_change)
        self.global_states.append(global_state)
        self.img_states.append(img_state)
        self.fact_states.append(fact_state)
        self.unit_states.append(unit_state)
        self.fact_actions.append(fact_action)
        self.unit_actions.append(unit_action)
        self.fact_probs.append(fact_prob)
        self.unit_probs.append(unit_prob)
        self.unit_probs_R.append(unit_prob_R)
        self.unit_probs_N.append(unit_prob_N)
        self.unit_probs_Rep.append(unit_prob_Rep)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        self.avail_facts.append(avail_facts)
        self.avail_units.append(avail_units)

    def clear_memory(self):
        self.global_states = []
        self.img_states = []
        self.fact_states = []
        self.unit_states = []
        self.fact_actions = []
        self.unit_actions = []
        self.fact_probs = []
        self.unit_probs = []
        self.unit_probs_R = []
        self.unit_probs_N = []
        self.unit_probs_Rep = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.avail_facts = []
        self.avail_units = []
