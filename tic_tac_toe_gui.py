import tkinter as tk
import random
import pickle
import os
import sys
from tic_tac_toe_env import TicTacToe
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource (works in dev and for PyInstaller .exe) """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

with open(resource_path("q_table_x.pkl"), "rb") as f:
    q_table_x = pickle.load(f)
with open(resource_path("q_table_o.pkl"), "rb") as f:
    q_table_o = pickle.load(f)

def get_q_value(q_table, state, action):
    return q_table.get((state, action), 0)

def select_action(q_table, state, available_actions):
    q_values = [get_q_value(q_table, state, a) for a in available_actions]
    return available_actions[np.argmax(q_values)] if q_values else random.choice(available_actions)

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe AI")
        self.env = TicTacToe()
        self.buttons = []
        self.message = tk.Label(self.root, text="", font=("Arial", 14))
        self.message.grid(row=3, column=0, columnspan=3)
        self.restart_button = None
        self.build_grid()
        self.game_over = False
        self.new_game()

    def build_grid(self):
        for i in range(9):
            btn = tk.Button(self.root, text="", font=("Arial", 24), width=5, height=2,
                            command=lambda i=i: self.on_click(i))
            btn.grid(row=i // 3, column=i % 3)
            self.buttons.append(btn)

    def new_game(self):
        self.env.reset()
        self.state, _ = self.env.reset()
        self.game_over = False
        self.player_turn = random.choice([True, False])
        self.message.config(text="Your turn!" if self.player_turn else "AI starts...")
        
        self.update_ui()

        if not self.player_turn:
            self.root.after(500, self.ai_move)

        if self.restart_button:
            self.restart_button.destroy()

    def update_ui(self):
        obs = np.array(self.env.state)
        for i in range(9):
            val = obs[i]
            text = "X" if val == 1 else "O" if val == -1 else ""
            self.buttons[i].config(text=text, state="normal" if text == "" and not self.game_over else "disabled")

    def on_click(self, index):
        if self.game_over or not self.player_turn or self.env.state[index] != 0:
            return
        self.play_move(index, player=True)
        if not self.game_over:
            self.root.after(500, self.ai_move)

    def ai_move(self):
        state_str = self.env._get_obs()
        available = self.env.get_available_actions()
        player = self.env.get_player_turn()
        q_table = q_table_x if player == 1 else q_table_o
        action = select_action(q_table, state_str, available)
        self.play_move(action, player=False)

    def play_move(self, index, player):
        _, reward, terminated, _, _ = self.env.step(index)
        self.update_ui()
        if terminated:
            self.game_over = True
            if reward == 0:
                msg = "It's a tie!"
            elif (reward == 1 and player) or (reward == -1 and not player):
                msg = "You win!" if player else "AI wins!"
            else:
                msg = "AI wins!" if player else "You win!"
            self.message.config(text=msg)
            self.restart_button = tk.Button(self.root, text="Play Again", font=("Arial", 12), command=self.new_game)
            self.restart_button.grid(row=4, column=0, columnspan=3)
        else:
            self.player_turn = not player

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
