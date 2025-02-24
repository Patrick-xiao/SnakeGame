import numpy as np
from collections import deque
import random


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = list(range(self.action_size))  # Define the action space
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997  # Slow down the decay rate
        self.learning_rate = 0.005  # Increase the learning rate
        self.gamma = 0.97  # Increase the consideration of long - term rewards
        self.memory = deque(maxlen=5000)  # Increase the memory capacity
        self.model = {}

    def get_q_values(self, state):
        state_key = tuple(np.round(state, 2))
        if state_key not in self.model:
            # Initialize with higher initial values for actions related to the food direction
            base = [random.uniform(-0.5, 0.5) for _ in range(self.action_size)]
            if state[0] > 0: base[1] += 0.5  # If the food is on the right, prefer to move right
            if state[0] < 0: base[3] += 0.5  # If the food is on the left, prefer to move left
            if state[1] > 0: base[2] += 0.5  # If the food is below, prefer to move down
            if state[1] < 0: base[0] += 0.5  # If the food is above, prefer to move up
            self.model[state_key] = np.array(base)
        return self.model[state_key]

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q = np.max(self.get_q_values(next_state))
                target = reward + self.gamma * next_q

            q_values = self.get_q_values(state)
            q_values[action] = q_values[action] * (1 - self.learning_rate) + target * self.learning_rate
            self.model[tuple(np.round(state, 2))] = q_values

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class SnakeEnv:
    def __init__(self, width=15, height=15):
        self.width = width
        self.height = height
        self.action_space = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.reset()

    def reset(self):
        self.snake = [(7, 7), (6, 7), (5, 7)]
        self.direction = 1
        self.foods = []
        self._generate_food(3)
        self.steps_since_eaten = 0
        return self._get_state()

    def _generate_food(self, num):
        while len(self.foods) < num:
            pos = (np.random.randint(1, self.width - 1),
                   np.random.randint(1, self.height - 1))
            if pos not in self.snake and pos not in self.foods:
                self.foods.append(pos)

    def _get_state(self):
        head = self.snake[0]
        nearest_food = min(self.foods, key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]))

        # Improved 8 - direction danger detection
        dangers = [
            (head[0], head[1] - 1) in self.snake or head[1] == 0,  # Up
            (head[0] + 1, head[1]) in self.snake or head[0] == self.width - 1,  # Right
            (head[0], head[1] + 1) in self.snake or head[1] == self.height - 1,  # Down
            (head[0] - 1, head[1]) in self.snake or head[0] == 0  # Left
        ]

        # Relative direction of the food (represented in polar coordinates)
        dx = nearest_food[0] - head[0]
        dy = nearest_food[1] - head[1]
        distance = np.sqrt(dx ** 2 + dy ** 2) + 1e-5
        food_dir = [dx / distance, dy / distance]

        # Current movement direction vector
        move_vec = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]

        return np.array([
            *food_dir,  # Food direction (2)
            *move_vec,  # Movement direction (2)
            distance / 20,  # Normalized distance (1)
            self.steps_since_eaten / 100,  # Hunger level (1)
            *[int(d) for d in dangers]  # Danger detection (4)
        ])

    def step(self, action):
        self.steps_since_eaten += 1
        head = self.snake[0]

        # Direction conversion limit
        if (action + self.direction) % 2 == 1:
            self.direction = action

        # Move
        move = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        new_head = (head[0] + move[0], head[1] + move[1])

        # Collision detection
        wall_collision = (new_head[0] < 0 or new_head[0] >= self.width or
                          new_head[1] < 0 or new_head[1] >= self.height)
        body_collision = new_head in self.snake
        dead = wall_collision or body_collision

        reward = 0

        # Basic survival reward (decays over time)
        reward += 0.1 * (1 - self.steps_since_eaten / 200)

        if dead:
            reward -= 20 + self.steps_since_eaten * 0.1  # Death penalty
            return self._get_state(), reward, True, {}

        # Calculate food attraction reward
        nearest_food = min(self.foods, key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]))
        old_dist = abs(head[0] - nearest_food[0]) + abs(head[1] - nearest_food[1])
        new_dist = abs(new_head[0] - nearest_food[0]) + abs(new_head[1] - nearest_food[1])

        # Progressive distance reward
        dist_reward = (old_dist - new_dist) * 2
        reward += dist_reward

        # Direction reward (the movement direction is consistent with the food direction)
        dx = nearest_food[0] - new_head[0]
        dy = nearest_food[1] - new_head[1]
        dir_vec = [dx / (abs(dx) + 1e-5), dy / (abs(dy) + 1e-5)]
        move_vec = move
        direction_reward = (dir_vec[0] * move_vec[0] + dir_vec[1] * move_vec[1]) * 0.5
        reward += direction_reward

        # Food eating detection
        if new_head in self.foods:
            self.foods.remove(new_head)
            self._generate_food(1)
            self.steps_since_eaten = 0
            reward += 50  # Significantly increase the food reward
            self.snake.insert(0, new_head)
            # Additional length reward
            reward += len(self.snake) * 0.5
        else:
            self.snake.pop()
            self.snake.insert(0, new_head)

        return self._get_state(), reward, False, {}


import tkinter as tk


class SnakeGame:
    def __init__(self, width=15, height=15, block_size=40):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.env = SnakeEnv(width, height)
        self.agent = QLearningAgent(state_size=10, action_size=4)

        # Initialize the GUI
        self.root = tk.Tk()
        self.root.title("AI Snake Game")
        self.root.configure(bg='#1a1a1a')

        # Game canvas
        self.canvas = tk.Canvas(
            self.root,
            width=width * block_size,
            height=height * block_size,
            bg='#2c3e50',
            highlightthickness=0
        )
        self.canvas.pack(pady=20, padx=20)

        # Information panel
        self.info_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.info_frame.pack()

        # Create labels
        self.labels = {
            'score': self.create_label("Score: 0", '#e74c3c'),
            'high_score': self.create_label("High Score: 0", '#3498db'),
            'epsilon': self.create_label("Exploration: 100%", '#9b59b6'),
            'avg_reward': self.create_label("Avg Reward: 0.0", '#2ecc71')
        }

        # Training parameters
        self.total_reward = 0
        self.episode = 0
        self.rewards = []
        self.high_score = 0

        # Start the game loop
        self.root.after(100, self.update)
        self.root.mainloop()

    def create_label(self, text, color):
        """Create an information label"""
        label = tk.Label(
            self.info_frame,
            text=text,
            font=('Roboto', 14, 'bold'),
            bg='#1a1a1a',
            fg=color
        )
        label.pack(side='left', padx=20)
        return label

    def draw_block(self, x, y, color):
        """Draw game elements"""
        return self.canvas.create_oval(
            x * self.block_size + 2,
            y * self.block_size + 2,
            (x + 1) * self.block_size - 2,
            (y + 1) * self.block_size - 2,
            fill=color,
            outline='' if color == '#e74c3c' else '#27ae60',
            width=2
        )

    def update(self):
        """Game main loop"""
        state = self.env._get_state()
        action = self.agent.act(state)
        next_state, reward, done, _ = self.env.step(action)

        # Record experience
        self.agent.remember(state, action, reward, next_state, done)
        self.agent.replay()

        # Update statistics
        self.total_reward += reward
        self.rewards.append(reward)

        # Update the display
        self.canvas.delete("all")

        # Draw the grid
        for x in range(self.width):
            for y in range(self.height):
                self.canvas.create_rectangle(
                    x * self.block_size,
                    y * self.block_size,
                    (x + 1) * self.block_size,
                    (y + 1) * self.block_size,
                    outline='#34495e',
                    width=1
                )

        # Draw the snake
        for i, (x, y) in enumerate(self.env.snake):
            # Gradient color
            color = '#%02x%02x%02x' % (
                46 + int(200 * i / len(self.env.snake)),
                204 - int(100 * i / len(self.env.snake)),
                113 - int(50 * i / len(self.env.snake))
            )
            self.draw_block(x, y, color)

        # Draw the food
        for food in self.env.foods:
            self.draw_block(*food, '#e74c3c')
            # Food halo
            self.canvas.create_oval(
                food[0] * self.block_size - 5,
                food[1] * self.block_size - 5,
                (food[0] + 1) * self.block_size + 5,
                (food[1] + 1) * self.block_size + 5,
                outline='#e67e22',
                width=2
            )

        # Update information
        current_score = len(self.env.snake) - 3
        if current_score > self.high_score:
            self.high_score = current_score

        self.labels['score'].config(text=f"Score: {current_score}")
        self.labels['high_score'].config(text=f"High Score: {self.high_score}")
        self.labels['epsilon'].config(
            text=f"Exploration: {self.agent.epsilon * 100:.1f}%"
        )

        # Calculate the average reward
        avg = np.mean(self.rewards[-100:]) if self.rewards else 0
        self.labels['avg_reward'].config(text=f"Avg Reward: {avg:.2f}")

        # Handle game over
        if done:
            self.episode += 1
            self.rewards.append(self.total_reward)
            self.total_reward = 0
            self.env.reset()

        # Adjust the game speed
        speed = 50 if self.agent.epsilon < 0.2 else 100
        self.root.after(speed, self.update)


if __name__ == "__main__":
    game = SnakeGame()