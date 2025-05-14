import random
from tic_tac_toe_env import TicTacToe
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np


#============================== Helpers ======================================================
# Helper function to get Q-value
def get_q_value(q_table, state, action):
    return q_table.get((state, action), 0)

# Helper function to update Q-value
def update_q_table(q_table, state, action, reward, next_state, next_actions, alpha, gamma):
    max_q_next = max([get_q_value(q_table, next_state, a) for a in next_actions], default=0)
    current_q = get_q_value(q_table, state, action)
    q_table[(state, action)] = current_q + alpha * (reward + gamma * max_q_next - current_q)

# Helper function to select an action using epsilon-greedy policy
def select_action(q_table, state, available_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(available_actions)
    q_values = [get_q_value(q_table, state, a) for a in available_actions]
    return available_actions[np.argmax(q_values)]
#=============================================================================================


#=================================== Model Parameters ========================================
N_GAMES       = 5_000  # Number of games for training
ALPHA         = 0.1  # Learning rate
GAMMA         = 0.9  # Discount factor
EPSILON       = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995  # Decay rate for epsilon
eval_interval = 25

# Model Variable Initialization
q_table_x = {}
q_table_o = {}
#=============================================================================================


#=================================== Environment Variables ===================================
env                  = TicTacToe()
draws, xWins, oWins  = [], [], []
drawCount, xWinCount = 0, 0
draw_games           = []
start_time           = dt.datetime.now()
verbose              = True
#=============================================================================================



#================================ Training Loop ==============================================

for game in range(N_GAMES):
    #------------------ Episode Reset -----------------------------------
    state, _ = env.reset()  # Reset the game
    terminated = False
    epsilon = EPSILON * (EPSILON_DECAY ** game)  # Decay epsilon
    x_moves = []  # Store X player's state-action pairs
    o_moves = []  # Store O player's state-action pairs

    while not terminated:
        player_turn = env.get_player_turn()
        available_actions = env.get_available_actions()

        #----------------- Choose Action ---------------------------------
        if player_turn == 1:  # X player's turn
            action = select_action(q_table_x, state, available_actions, epsilon)
            x_moves.append((state, action))
        else:  # O player's turn
            action = select_action(q_table_o, state, available_actions, epsilon)
            o_moves.append((state, action))

        #---------- Take action in environment --------------------------
        next_state, reward, terminated, _, _ = env.step(action)
        next_actions = env.get_available_actions()

        #----------- Game end logic --------------------------------------
        if terminated:
            # Set specific reward for Xs and Os
            if reward == 0:  # Draw
                x_reward, o_reward = 0, 0
                drawCount += 1
            else:  # Win or lose
                x_reward = reward if player_turn == 1 else -reward
                o_reward = -reward if player_turn == 1 else reward

            # Graph Management 
            if player_turn == 1 and reward == 1:
                xWinCount += 1
            draw_games.append(reward == 0)

            # Update Q values for all moves 
            for s, a in x_moves:
                update_q_table(q_table_x, s, a, x_reward, next_state, next_actions, ALPHA, GAMMA)
            for s, a in o_moves:
                update_q_table(q_table_o, s, a, o_reward, next_state, next_actions, ALPHA, GAMMA)

        else:
            state = next_state  # Update the current state

    #------------ Plotting and Printing -------------------------------------
    if (game+1) % eval_interval == 0:
        draws.append(drawCount / eval_interval)
        xWins.append(xWinCount / eval_interval)
        oWins.append((eval_interval - (xWinCount + drawCount)) / eval_interval)
        drawCount, xWinCount = 0, 0
        if verbose: print(f'Completed step {game+1}')


#================== Record Draws to CSV ===========================================
pd.Series(draw_games).to_csv('draw_games.csv')


#======================= Plot Results =============================================
if verbose:
    end_time = dt.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time}.")

    # X wins per eval interval: blue
    plt.plot(xWins)
    # O wins per eval interval: orange
    plt.plot(oWins)
    # Draws per eval interval:  green
    plt.plot(draws)
    plt.show()

