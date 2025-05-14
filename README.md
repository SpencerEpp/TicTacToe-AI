# Tic-Tac-Toe Reinforcement Learning Agent
This project implements a reinforcement learning agent that learns to play Tic-Tac-Toe through self-play using Q-learning. The training is performed in a custom OpenAI Gym-style environment.


## Overview
The environment simulates a standard 3×3 Tic-Tac-Toe board. Two agents — X and O — alternate turns. The game ends with a win, draw, or full board. A Q-learning algorithm is used to train both agents, with an epsilon-greedy policy to balance exploration and exploitation.


## Key Features
- Self-play between two Q-learning agents  
- Reward tuning for win/loss/draw outcomes  
- Epsilon decay for exploration control  
- Game statistics tracking and visualization  
- Gym-style interface for easy extensibility  


## File Structure
| File                   | Description |
|------------------------|-------------|
| `tic_tac_toe_env.py`   | Custom Gym-compatible Tic-Tac-Toe environment with board logic, rendering, and reward handling |
| `tic_tac_toe_solution.py` | Main training script implementing Q-learning for both players with data collection and plotting |
| `draw_games.csv`       | (Generated) CSV file logging whether each game ended in a draw |
| `README.md`            | Project documentation (this file) |


## How It Works
- Each game is reset using the environment’s `reset()` method.  
- At each turn, the agent chooses an action based on an epsilon-greedy strategy.  
- The environment returns the next state, reward, and a terminal flag.  
- Q-tables for both players (X and O) are updated after each move.  
- Rewards are:  
  - `+1` for a win  
  - `-1` for a loss  
  - `0` for a draw or intermediate move  
- After training, win/draw statistics are saved and plotted.


## Running the Project
To start training: python tic_tac_toe_solution.py
This will:
- Train both agents over 5,000 games
- Save draw statistics to draw_games.csv
- Display plots of win/draw rates over time


## Output
- Console logs: Training progress every 25 games
- CSV: draw_games.csv shows a boolean per game indicating a draw
- Plots: Line graphs of win and draw rates over training intervals


## Customization
You can adjust training parameters in tic_tac_toe_solution.py:
- N_GAMES: Total games for training
- ALPHA: Learning rate
- GAMMA: Discount factor
- EPSILON, EPSILON_DECAY: Exploration behavior