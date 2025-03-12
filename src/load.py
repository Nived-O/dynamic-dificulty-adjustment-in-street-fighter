import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Round
import pygame
from pygame.locals import KEYDOWN
from game import Point
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
import menu



class StopTrainingOnDoneCallback(BaseCallback):
    def _init_(self, verbose=0):
        super()._init_(verbose)

    def _on_step(self) -> bool:
        done = self.locals["dones"]  # Get the 'done' flag
        if any(done):
            print("Stopping training because 'done' is True.")
            return False  # Stop training immediately
        return True  # Continue training

def log_data_with_actions(observation, player1_action, player2_action):
    row = observation + [player1_action, player2_action]
    with open("data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)

class StreetFighterEnv(gym.Env):
    def __init__(self):
        super(StreetFighterEnv, self).__init__()
        
        # Observation space: Positions and health normalized between 0 and 1
        # 6 values: x1, y1, health1, x2, y2, health2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),  # 8 features: normalized positions,health and energy
            dtype=np.float32
        )
        
        # Action space: 7 discrete actions (4 movements + 2 attacks + 1 special)
        self.action_space = spaces.Discrete(8)
        
        # Game-specific attributes

        # pygame.init()
        # pygame.mixer.init()
        self.screen_width = 320
        self.screen_height = 240
        # self.screen = pygame.display.set_mode((320, 240), 0, 32)
        # self.player1 = Round.Player('Ken', 120, 100)
        # self.player2 = Round.Player('Ken', 120, 100, Player2=True,alt_color = True)
        # print('loading backgRound...')
        # self.backgRound = Round.Background('../res/BackgRound/Bckgrnd0.png')
        # print('creating game...')
        # game = Round.Game(self.screen,self.backgRound,self.player1,self.player2)
        

        
    def reset(self,seed=None):
        """Reset the environment to its initial state."""
        # Reset player states (example logic)
        ###pygame.init()
        ###pygame.mixer.init()
        ###config.Screen()
        ###self.screen = pygame.Surface((320, 240), 0, 32)
        self.screen = pygame.display.set_mode((650, 500), 0, 32)
        self.player1 = Round.Player('Ken', 120, 100)
        self.player2 = Round.Player('Ken', 120, 100, Player2=True,alt_color = True)
        print('loading backgRound...')
        self.backgRound = Round.Background('../res/BackgRound/Bckgrnd0.png')
        print('creating game...')
        self.game = Round.Game(self.screen,self.backgRound,self.player1,self.player2)
        Round.round_marker = (False, False)
        Round.end_round = False
        Round.ready_text = menu.Text('Ready',Point(128,120))
        Round.fight_text = menu.Text('Fight !!',Point(96,120))
        
        
        # Return the initial observation
        return self._get_observation(),{}
    #,self.game
    
    def step(self, action):
        """Execute one step in the environment."""
        # Apply the chosen action (implement your own logic here)
        #self._apply_action(self.player1, action)
        
        # Update the game state (you may have a game loop or logic to handle       
        # Get the current observation
        i1=self.game.mainloop(action)
        observation = self._get_observation()
        reward=0
        # Calculate the reward (implement your reward logic)
        #reward = self._calculate_reward()
        if action == 7:
            reward=1
        if i1 !=None:
            reward+=5
        if(action in(4,5,6)):
            reward+=1
        if(i1==None and action in(4,5,6)):
            reward+=-2
        
        # Determine if the game is done
        done = self._is_done()
        if done:
            if(self.player1.health.hp<1):
                self.won=2
            else:
                self.won=1
        info={}
        truncated = False
        
        return observation, reward, done,truncated, info
    
    def _get_observation(self):
        """Generate the observation for the current state."""
        return np.array([
            self.player1.position.x / self.screen_width,
            self.player1.position.y / self.screen_height,
            self.player1.health.hp / 1000,  # Normalized health1
            self.player1.energy.energy / 96,
            self.player2.position.x / self.screen_width,
            self.player2.position.y / self.screen_height,
            self.player2.health.hp / 1000,   # Normalized health2
            self.player2.energy.energy / 96
        ], dtype=np.float32)
    

    
    # def _update_game_state(self):
    #     """Update the state of the game (e.g., handle collisions, time)."""
    #     pass  # Implement game logic
    
    def _calculate_reward(self):
        """Calculate the reward for the current state."""
        # Example: Reward based on damage dealt to the opponent
        damage_dealt = 1000 - self.player2.health.hp
        return damage_dealt
    
    def _is_done(self):
        """Check if the game is over."""
        # Example: Game ends if one player's health reaches 0
        return Round.end_round



def rein():
    env=StreetFighterEnv()
    model = PPO.load("ppo_streetfighter", env=env)
    model.learn(total_timesteps=100,callback=StopTrainingOnDoneCallback())
    model.save("ppo_streetfighter")
    ppo_model = PPO.load("ppo_streetfighter")



    env.screen.fill((0, 0, 0))  # RGB for white
    pygame.display.flip()
    if(env.won==2):
        WIDTH, HEIGHT = 100, 10
        BAR_WIDTH = 100
        BAR_HEIGHT = 30
        BAR_X = (WIDTH - BAR_WIDTH) // 2
        BAR_Y = (HEIGHT - BAR_HEIGHT) // 2
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)  
        pygame.draw.rect(env.screen, WHITE, (BAR_X, BAR_Y, BAR_WIDTH, BAR_HEIGHT), 2)
        # Load observation & action datasets
        obs_data = torch.load("fighting_data.pt")   # Observations (states)
        action_data = torch.load("action_data.pt")  # Actions

        # Ensure correct tensor types
        obs_data = obs_data.float()        # Observations should be float32
        action_data = action_data.long()   # Actions should be int64 for CrossEntropyLoss

        # Define loss function & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(ppo_model.policy.parameters(), lr=1e-4)

        # Get number of possible actions
        num_classes = 8  # Adjust if needed

        # Training loop
        num_epochs = 5
        for epoch in range(num_epochs):
            for obs, action in zip(obs_data, action_data):
                obs = obs.unsqueeze(0)  # Ensure it has a batch dimension
                action = action.unsqueeze(0)  # Ensure it has a batch dimension
                distribution = ppo_model.policy.get_distribution(obs)

                # Extract logits manually
                predicted_action_logits = distribution.distribution.logits  # SB3 stores it inside `.distribution`
                # Ensure action is a single index (not one-hot, not float)
                action = action.long().view(-1)  # Shape: [1]

                # Compute loss
                loss = criterion(predicted_action_logits, action)   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        ppo_model.save("ppo_streetfighter")
        print("✅ Training complete! Model saved as ppo_streetfighter_bc")
        with open("../res/level.txt", "r") as file:
            level = file.read().strip()
            new_level = str(int(level) + 1)
            with open("../res/level.txt", "w") as file:
                file.write(new_level)


if __name__ == "__main__":
    ###options = config.OptionConfig() 
    ###pygame.init()
    ###pygame.mixer.init()
    ###config.Screen()   
    screen = pygame.display.set_mode((640, 480))
    menu = menu.MainMenu(screen, Point(50,140))
    menu.mainloop()
    print("fkisgd fhsadofihdesiof hoisuadh fkish fiksdbh fishdoifhsdoifh siohf iusdhf uishfihs ifhs iof")
    env=StreetFighterEnv()
    model = PPO.load("ppo_streetfighter", env=env)
    model.learn(total_timesteps=10)
    model.save("ppo_streetfighter")





    ppo_model = PPO.load("ppo_streetfighter1")

    # Load observation & action datasets
    obs_data = torch.load("fighting_data.pt")   # Observations (states)
    action_data = torch.load("action_data.pt")  # Actions

    # Ensure correct tensor types
    obs_data = obs_data.float()        # Observations should be float32
    action_data = action_data.long()   # Actions should be int64 for CrossEntropyLoss

    # Define loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ppo_model.policy.parameters(), lr=1e-4)

    # Get number of possible actions
    num_classes = 8  # Adjust if needed

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for obs, action in zip(obs_data, action_data):
            obs = obs.unsqueeze(0)  # Ensure it has a batch dimension
            action = action.unsqueeze(0)  # Ensure it has a batch dimension
            distribution = ppo_model.policy.get_distribution(obs)

            # Extract logits manually
            predicted_action_logits = distribution.distribution.logits  # SB3 stores it inside `.distribution`
            # Ensure action is a single index (not one-hot, not float)
            action = action.long().view(-1)  # Shape: [1]

            # Compute loss
            loss = criterion(predicted_action_logits, action)   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    ppo_model.save("ppo_streetfighter_bc")
    print("✅ Training complete! Model saved as ppo_streetfighter_bc")

