import streamlit as st
import numpy as np
import pandas as pd
import time

# ---------- Grid Definition ----------
H, W = 4, 4

START = (0, 0)
GOAL = (2, 3)
PIT = (3, 2)
WALL = (1, 1)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["↑", "↓", "←", "→"]

STEP_COST = -0.1
GOAL_REWARD = 10
PIT_REWARD = -10

# ---------- Helpers ----------
def is_valid(pos):
    r, c = pos
    if r < 0 or r >= H or c < 0 or c >= W:
        return False
    if pos == WALL:
        return False
    return True

def get_reward(pos):
    if pos == GOAL:
        return GOAL_REWARD
    if pos == PIT:
        return PIT_REWARD
    return STEP_COST

def get_next_states(r, c, action, p_success):
    outcomes = []

    intended = action
    others = [a for a in ACTIONS if a != action]

    moves = [(intended, p_success)]
    slip_prob = (1 - p_success) / 3

    for a in others:
        moves.append((a, slip_prob))

    for a, p in moves:
        nr, nc = r + a[0], c + a[1]
        if not is_valid((nr, nc)):
            nr, nc = r, c
        outcomes.append((nr, nc, p))

    return outcomes

def sample_next_state(r, c, action, p_success):
    outcomes = get_next_states(r, c, action, p_success)
    probs = [p for _, _, p in outcomes]
    idx = np.random.choice(len(outcomes), p=probs)
    return outcomes[idx][0], outcomes[idx][1]

# ---------- Value Iteration ----------
def value_iteration(gamma, p_success, theta=1e-4):
    V = np.zeros((H, W))

    while True:
        delta = 0
        for r in range(H):
            for c in range(W):
                s = (r, c)
                if s in [GOAL, PIT, WALL]:
                    continue

                v_old = V[r, c]
                action_values = []

                for a in ACTIONS:
                    total = 0
                    for nr, nc, p in get_next_states(r, c, a, p_success):
                        reward = get_reward((nr, nc))
                        total += p * (reward + gamma * V[nr, nc])
                    action_values.append(total)

                V[r, c] = max(action_values)
                delta = max(delta, abs(v_old - V[r, c]))

        if delta < theta:
            break

    return V

def extract_policy(V, gamma, p_success):
    policy = [[" " for _ in range(W)] for _ in range(H)]

    for r in range(H):
        for c in range(W):
            s = (r, c)

            if s == WALL:
                policy[r][c] = "X"
                continue
            if s == GOAL:
                policy[r][c] = "G"
                continue
            if s == PIT:
                policy[r][c] = "P"
                continue

            best_v = -1e9
            best_a = None

            for i, a in enumerate(ACTIONS):
                total = 0
                for nr, nc, p in get_next_states(r, c, a, p_success):
                    reward = get_reward((nr, nc))
                    total += p * (reward + gamma * V[nr, nc])

                if total > best_v:
                    best_v = total
                    best_a = ACTION_NAMES[i]

            policy[r][c] = best_a

    return policy

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RL Gridworld Demo", layout="wide")
st.title("RL Gridworld with Live Agent")

with st.sidebar:
    st.header("Parameters")
    gamma = st.slider("Discount Factor (γ)", 0.0, 0.99, 0.9, 0.01)
    p_success = st.slider("Transition Success Probability", 0.25, 1.0, 0.8, 0.05)

    st.markdown("---")
    run_sim = st.button("▶ Run Agent")
    reset_sim = st.button("🔄 Reset Agent")

# ---------- RL Solve ----------
V = value_iteration(gamma, p_success)
policy = extract_policy(V, gamma, p_success)

# ---------- Session State for Agent ----------
if "agent_pos" not in st.session_state or reset_sim:
    st.session_state.agent_pos = START

# ---------- Display Tables ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Value Function V(s)")
    st.dataframe(pd.DataFrame(np.round(V, 2)), use_container_width=True)

with col2:
    st.subheader("Optimal Policy π(s)")
    st.dataframe(pd.DataFrame(policy), use_container_width=True)

# ---------- Grid with Agent ----------
with col3:
    st.subheader("Grid World (Live Agent)")

    grid = [["." for _ in range(W)] for _ in range(H)]

    for r in range(H):
        for c in range(W):
            if (r, c) == WALL:
                grid[r][c] = "X"
            elif (r, c) == GOAL:
                grid[r][c] = "G"
            elif (r, c) == PIT:
                grid[r][c] = "P"

    ar, ac = st.session_state.agent_pos
    grid[ar][ac] = "🤖"

    st.dataframe(pd.DataFrame(grid), use_container_width=True)

# ---------- Simulation Step ----------
if run_sim:
    ar, ac = st.session_state.agent_pos

    if (ar, ac) not in [GOAL, PIT]:
        action_symbol = policy[ar][ac]
        action_map = {"↑": (-1, 0), "↓": (1, 0), "←": (0, -1), "→": (0, 1)}

        if action_symbol in action_map:
            a = action_map[action_symbol]
            nr, nc = sample_next_state(ar, ac, a, p_success)
            st.session_state.agent_pos = (nr, nc)

        time.sleep(0.4)
        st.rerun()

# ---------- Teaching Notes ----------
st.markdown("### Teaching Intuition")
st.markdown(f"""
- **γ = {gamma}** controls patience  
- **Transition success = {p_success}** controls uncertainty  
- Agent follows optimal policy but may slip  
- Watch how behavior changes with parameters  
""")
