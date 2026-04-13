"""Test harness for cartpole_rl — imports CartPoleAgent from solution.py"""
import numpy as np

np.random.seed(42)
from solution import CartPoleAgent

passed = 0
total = 2


class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self.state = None

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = bool(abs(x) > 2.4 or abs(theta) > 0.209)
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done


# Train
env = CartPoleEnv()
agent = CartPoleAgent()

for ep in range(5000):
    state = env.reset()
    for step in range(200):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

# Evaluate
rewards = []
old_eps = agent.epsilon
agent.epsilon = 0.0
for _ in range(100):
    state = env.reset()
    total_reward = 0
    for step in range(200):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    rewards.append(total_reward)
agent.epsilon = old_eps

avg_reward = np.mean(rewards)
if avg_reward >= 50:
    passed += 1
    print(f"PASS test_reward (avg={avg_reward:.1f})")
else:
    print(f"FAIL test_reward (avg={avg_reward:.1f}, need >= 50)")

# Test 2: q_table attribute
if hasattr(agent, "q_table"):
    passed += 1
    print("PASS test_q_table_exists")
else:
    print("FAIL test_q_table_exists")

print(f"RESULT {passed}/{total}")
