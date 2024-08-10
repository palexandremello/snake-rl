import random
import pygame
import numpy as np
import sys
from enum import Enum

# Actions the Snake can perform
class SnakeAction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

class SnakeGame:
    def __init__(self, grid_width=32, grid_height=32, block_size=10, fps=15):
        self.grid_width = grid_width * block_size
        self.grid_height = grid_height * block_size
        self.block_size = block_size
        self.fps = fps
        self.score = 0
        self.game_over = False
        self.got_apple = False
        self.snake_head_distance_to_apple = 0

        self.reset()
        self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()

        self.clock = pygame.time.Clock()

        self.window_size = (self.grid_width, self.grid_height)
        self.window_surface = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Snake Game")

        self.font = pygame.font.SysFont(None, 35)

    def reset(self):
        self.pos_snake_x = self.grid_width // 2
        self.pos_snake_y = self.grid_height // 2
        self.snake_segments = [
            (self.pos_snake_x, self.pos_snake_y),
            (self.pos_snake_x - self.block_size, self.pos_snake_y),
            (self.pos_snake_x - 2 * self.block_size, self.pos_snake_y),
        ]

        self.pos_apple_x, self.pos_apple_y = self._get_apple_position_near_snake()
        head_x, head_y = self.snake_segments[0]

        self.snake_head_distance_to_apple = self._calculate_distance(
            head_x, head_y, self.pos_apple_x, self.pos_apple_y
        )
        self.dir_snake_x = self.block_size
        self.dir_snake_y = 0
        self.last_dir_x = self.dir_snake_x
        self.last_dir_y = self.dir_snake_y
        self.score = 0
        self.game_over = False

    def _calculate_distance(self, x1, y1, x2, y2):
        return (((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5) / self.block_size

    def _get_apple_position_near_snake(self):
        max_distance = (
            self.block_size * 5
        )  # Define a maximum distance from the snake's head
        head_x, head_y = self.snake_segments[0]

        while True:
            pos_apple_x = (
                round(
                    random.randrange(
                        max(head_x - max_distance, 0),
                        min(head_x + max_distance, self.grid_width - self.block_size),
                    )
                    / self.block_size
                )
                * self.block_size
            )
            pos_apple_y = (
                round(
                    random.randrange(
                        max(head_y - max_distance, 0),
                        min(head_y + max_distance, self.grid_height - self.block_size),
                    )
                    / self.block_size
                )
                * self.block_size
            )

            if (pos_apple_x, pos_apple_y) not in self.snake_segments:
                return pos_apple_x, pos_apple_y

    def step(self, action: SnakeAction):
        if action == SnakeAction.LEFT and self.last_dir_x != self.block_size:
            self.dir_snake_x = -self.block_size
            self.dir_snake_y = 0
        elif action == SnakeAction.RIGHT and self.last_dir_x != -self.block_size:
            self.dir_snake_x = self.block_size
            self.dir_snake_y = 0
        elif action == SnakeAction.UP and self.last_dir_y != self.block_size:
            self.dir_snake_x = 0
            self.dir_snake_y = -self.block_size
        elif action == SnakeAction.DOWN and self.last_dir_y != -self.block_size:
            self.dir_snake_x = 0
            self.dir_snake_y = self.block_size

        self.pos_snake_x += self.dir_snake_x
        self.pos_snake_y += self.dir_snake_y

        self.last_dir_x = self.dir_snake_x
        self.last_dir_y = self.dir_snake_y

        head_x, head_y = self.snake_segments[0]

        self.snake_head_distance_to_apple = self._calculate_distance(
            head_x, head_y, self.pos_apple_x, self.pos_apple_y
        )

        self.snake_segments.insert(0, (self.pos_snake_x, self.pos_snake_y))

        self.got_apple = False
        if (
            self.pos_snake_x == self.pos_apple_x
            and self.pos_snake_y == self.pos_apple_y
        ):
            self.pos_apple_x, self.pos_apple_y = self._get_apple_position_near_snake()
            self.score += 1
            self.got_apple = True
        else:
            self.snake_segments.pop()

        if (
            self.pos_snake_x < 0
            or self.pos_snake_x >= self.grid_width
            or self.pos_snake_y < 0
            or self.pos_snake_y >= self.grid_height
        ):
            self.game_over = True

        for segment in self.snake_segments[1:]:
            if self.pos_snake_x == segment[0] and self.pos_snake_y == segment[1]:
                self.game_over = True

    def render(self):
        self._process_events()

        self.window_surface.fill((160, 160, 160))

        for segment in self.snake_segments:
            pygame.draw.rect(
                self.window_surface,
                (0, 128, 0),
                [segment[0], segment[1], self.block_size, self.block_size],
            )

        pygame.draw.rect(
            self.window_surface,
            (255, 0, 0),
            [self.pos_apple_x, self.pos_apple_y, self.block_size, self.block_size],
        )

        # self.show_score()
        pygame.display.update()
        self.clock.tick(self.fps)

    def show_score(self):
        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window_surface.blit(text, [0, 0])

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

    def get_screen(self):
        screen = pygame.surfarray.array3d(self.window_surface)
        return np.transpose(screen, (1, 0, 2))

    def get_snake_head(self):
        return self.snake_segments[0]

    def get_apple_position(self):
        return (self.pos_apple_x, self.pos_apple_y)

    def get_observation(self):
        apple_direction = self._get_apple_direction()
        danger = self._get_danger()

        observation = [
            int(apple_direction["left"]),
            int(apple_direction["right"]),
            int(apple_direction["down"]),
            int(apple_direction["up"]),
            int(danger["down"]),
            int(danger["up"]),
            int(danger["left"]),
            int(danger["right"]),
        ]
        return observation

    def _get_apple_direction(self):
        head_x, head_y = self.snake_segments[0]
        apple_direction = {
            "left": head_x > self.pos_apple_x,
            "right": head_x < self.pos_apple_x,
            "up": head_y > self.pos_apple_y,
            "down": head_y < self.pos_apple_y,
        }
        return apple_direction

    def _get_danger(self):
        danger = {
            "left": self._is_danger(self.pos_snake_x - self.block_size, self.pos_snake_y),
            "right": self._is_danger(self.pos_snake_x + self.block_size, self.pos_snake_y),
            "up": self._is_danger(self.pos_snake_x, self.pos_snake_y - self.block_size),
            "down": self._is_danger(self.pos_snake_x, self.pos_snake_y + self.block_size),
        }
        return danger

    def _is_danger(self, x, y):
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if (x, y) in self.snake_segments:
            return True
        return False

if __name__ == "__main__":
    snake_game = SnakeGame()
    snake_game.render()

    while not snake_game.game_over:
        action = random.choice(list(SnakeAction))
        observation = snake_game.get_observation()
        print("Observation: ", observation)
        snake_game.step(action)
        snake_game.render()
