import numpy as np
import pygame
import random
import math

from pygame import gfxdraw

class GameEnvironment:
    def __init__(self, AI=True):
        self.player_radius = 20
        self.player_z = 0
        self.ROAD_DEPTH = 4000
        self.player_lateral_speed = 20
        self.ROAD_WIDTH = 1000
        self.OBSTACLE_HEIGHT = 40
        self.OBSTACLE_LENGTH = 10
        self.OBSTACLE_WIDTH = 40
        self.GRAVITY = 0.7
        self.AI = AI
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.offset_x, self.offset_y = 800 if AI else 0, 0
        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF  # Enable hardware acceleration and double buffering
        )
        pygame.display.set_caption("Pluck Stars - AI Control" if AI else "Pluck Stars - Player Control")
        self.clock = pygame.time.Clock()

        self.reset()
        self.draw()

    def reset(self):
        self.player_x = 0
        self.player_y = 0
        self.player_velocity_y = 14.0
        self.score = 0
        self.obstacles = []
        self.spawn_obstacle()

        return self.get_state()

    def spawn_obstacle(self):
        # Generate the first obstacle if none exist
        if len(self.obstacles) == 0:
            self.obstacles.append({
                "x": random.randrange(-self.ROAD_WIDTH // 2 + self.OBSTACLE_WIDTH // 2 + 30, -self.ROAD_WIDTH // 2 + self.OBSTACLE_WIDTH // 2 + 60, 20),
                "y": 100,
                "z": self.ROAD_DEPTH - 600,
                "height": self.OBSTACLE_HEIGHT,
                "get_score": False
            })

        # ==== Z-axis generation rules ====
        # Get the tail position of the previous obstacle (smaller z-coordinate means closer to the player)
        last_obstacle = self.obstacles[-1]
        last_z = last_obstacle["z"]
        last_x = last_obstacle["x"]

        # Minimum allowed position for the head of the new obstacle = tail of the previous obstacle + minimum distance
        min_z = min(last_z + 500, self.ROAD_DEPTH - 100)
        # Maximum allowed position for the head of the new obstacle = tail of the previous obstacle + maximum distance
        max_z = min(last_z + 1500, self.ROAD_DEPTH - 100)

        new_z = random.randrange(min_z, max_z + 1, 10)
        # ==== X-axis generation rules ====
        new_x = random.randrange(max(-self.ROAD_WIDTH // 2 + self.OBSTACLE_WIDTH // 2 + 30, last_x - 300), min(self.ROAD_WIDTH // 2 - self.OBSTACLE_WIDTH // 2 - 30, last_x + 300) + 1, 20)

        # ==== Generate new obstacle ====
        new_obstacle = {
            "x": new_x,
            "y": 100,  # Obstacle bottom height
            "z": new_z,  # Initial position of the obstacle head
            "height": self.OBSTACLE_HEIGHT,
            "get_score": False
        }
        self.obstacles.append(new_obstacle)

    def project_3d_to_2d(self, x, y, z):
        """Projection function with overhead effect"""
        rel_z = z - (self.player_z - 500)  # Correct rel_z calculation considering camera distance
        min_z = 1
        effective_z = max(rel_z, min_z)

        fov = math.radians(60)
        scale = self.WIDTH / (2 * math.tan(fov))

        # Add overhead offset
        y_offset = 200 + math.tan(math.radians(30)) * rel_z

        screen_x = (x * scale / effective_z) + self.WIDTH // 2
        screen_y = self.HEIGHT // 2 - ((y - y_offset) * scale / effective_z)

        return screen_x, screen_y

    def draw_player(self):
        """Draw the player ball with adjusted transparency"""
        player_screen_pos = self.project_3d_to_2d(0, self.player_y, self.player_radius)
        x, y = int(player_screen_pos[0]), int(player_screen_pos[1])

        # Ground shadow parameters (vary with height)
        shadow_alpha = max(100 - int(self.player_y * 0.7), 0)
        shadow_scale = 1 - self.player_y / 200
        shadow_width = int(self.player_radius * 1.2 * shadow_scale)
        shadow_height = int(self.player_radius * 0.6 * shadow_scale)

        # Draw ground shadow (ellipse)
        if shadow_width > 0 and shadow_height > 0:
            shadow_surface = pygame.Surface((shadow_width * 2, shadow_height * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, (50, 50, 50, shadow_alpha),
                                (0, 0, shadow_width * 2, shadow_height * 2))
            self.screen.blit(shadow_surface, (x - shadow_width, y + self.player_radius - 10))

        # Create a surface with an alpha channel for the ball
        ball_surface = pygame.Surface((self.player_radius * 2, self.player_radius * 2), pygame.SRCALPHA)

        # Draw the ball on the surface
        pygame.gfxdraw.filled_circle(ball_surface, self.player_radius, self.player_radius, self.player_radius,
                                     (200, 50, 50, 255))  # Less transparent red

        # Draw gradient layers for the ball
        for i in range(3):
            alpha = 255 - i * 20  # Adjusted transparency
            radius = self.player_radius - i * 3
            color = (255, 50 + i * 30, 50 + i * 30, alpha)
            pygame.gfxdraw.aacircle(ball_surface, self.player_radius, self.player_radius, radius, color)
            pygame.gfxdraw.filled_circle(ball_surface, self.player_radius, self.player_radius, radius, color)

        # Draw the highlight
        highlight_radius = self.player_radius // 3
        angle = math.radians(-45)  # Light source from top-right
        highlight_x = self.player_radius + int(self.player_radius * 0.6 * math.cos(angle))
        highlight_y = self.player_radius + int(self.player_radius * 0.6 * math.sin(angle))
        pygame.gfxdraw.filled_circle(ball_surface, highlight_x, highlight_y, highlight_radius,
                                     (255, 255, 255, 180))  # Less transparent white

        # Blit the ball surface onto the main screen
        self.screen.blit(ball_surface, (x - self.player_radius, y - self.player_radius))

    def draw_road(self):
        """Dynamic ground generation (covering the entire road depth)"""
        segment_depth = 50
        start_z = self.player_z - 500  # Start generating from the camera position
        end_z = start_z + self.ROAD_DEPTH  # End position
        num_segments = int((end_z - start_z) / segment_depth) + 1  # Calculate total number of segments

        for i in range(num_segments):
            z_start = start_z + i * segment_depth
            z_end = z_start + segment_depth

            # Ground vertex projection
            left_start = self.project_3d_to_2d(-self.ROAD_WIDTH / 2 - self.player_x, 0, z_start)
            right_start = self.project_3d_to_2d(self.ROAD_WIDTH / 2 - self.player_x, 0, z_start)
            left_end = self.project_3d_to_2d(-self.ROAD_WIDTH / 2 - self.player_x, 0, z_end)
            right_end = self.project_3d_to_2d(self.ROAD_WIDTH / 2 - self.player_x, 0, z_end)

            # Filter segments behind the camera
            if z_end <= self.player_z - 500:
                continue

            # Draw the ground
            pygame.draw.polygon(self.screen, (192, 192, 192), [
                (left_start[0], left_start[1]),
                (right_start[0], right_start[1]),
                (right_end[0], right_end[1]),
                (left_end[0], left_end[1])
            ], 0)

            # Draw road edges
            pygame.draw.line(self.screen, (0, 0, 0), left_start, left_end, 2)
            pygame.draw.line(self.screen, (0, 0, 0), right_start, right_end, 2)

        # Draw obstacle disappearance line (red horizontal line)
        left_disappear = self.project_3d_to_2d(-self.ROAD_WIDTH / 2 - self.player_x, 0, -self.OBSTACLE_LENGTH)
        right_disappear = self.project_3d_to_2d(self.ROAD_WIDTH / 2 - self.player_x, 0, -self.OBSTACLE_LENGTH)

        # Draw red horizontal line
        pygame.draw.line(self.screen, (255, 0, 0), left_disappear, right_disappear, 2)

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            x = obstacle["x"] - self.player_x
            z = obstacle["z"]
            y = obstacle["y"]
            height = obstacle["height"]  # Height of the star
            thickness = self.OBSTACLE_LENGTH  # Thickness of the star

            perspective_scale = 1 / (1 + z * 0.008)  # Coefficient 0.008 controls perspective strength

            base_size = self.OBSTACLE_WIDTH * perspective_scale
            shadow_width = int(base_size * 0.5)
            shadow_height = int(base_size * 0.3)

            # Calculate shadow center point (projection position at y=0)
            shadow_3d_pos = (x, 0, z + thickness / 2)  # Shadow is located directly below the center of the star
            shadow_screen_pos = self.project_3d_to_2d(*shadow_3d_pos)

            # Create a semi-transparent shadow surface
            shadow_surface = pygame.Surface((shadow_width * 2, shadow_height * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, (50, 50, 50, 80),  # 80 is the transparency
                                (0, 0, shadow_width * 2, shadow_height * 2))

            # Adjust drawing coordinates based on projection position
            self.screen.blit(shadow_surface,
                             (shadow_screen_pos[0] - shadow_width,
                              shadow_screen_pos[1] - shadow_height // 2))

            # Calculate the radii of the circumscribed and inscribed circles of the star
            outer_radius = self.OBSTACLE_WIDTH / 2
            inner_radius = outer_radius * (3 - math.sqrt(5)) / 2  # Inscribed radius

            # Generate the vertices of the star
            points = []
            for i in range(5):
                # Outer vertex
                angle = math.radians(72 * i)
                px = x + outer_radius * math.cos(angle)
                py = y + outer_radius * math.sin(angle)
                points.append((px, py, z))  # Front surface
                points.append((px, py, z + thickness))  # Back surface

                # Inner vertex
                angle_inner = math.radians(72 * i + 36)
                px_inner = x + inner_radius * math.cos(angle_inner)
                py_inner = y + inner_radius * math.sin(angle_inner)
                points.append((px_inner, py_inner, z))  # Front surface
                points.append((px_inner, py_inner, z + thickness))  # Back surface

            # Draw the star, yellow
            pygame.draw.polygon(self.screen, (255, 255, 0),
                                [self.project_3d_to_2d(px, py, pz) for px, py, pz in points[::2]])
            pygame.draw.polygon(self.screen, (255, 255, 0),
                                [self.project_3d_to_2d(px, py, pz) for px, py, pz in points[1::2]])

            # Draw edge lines
            for i in range(0, len(points), 2):
                front_point = points[i]
                back_point = points[i + 1]
                pygame.draw.line(self.screen, (255, 255, 255),
                                 self.project_3d_to_2d(*front_point),
                                 self.project_3d_to_2d(*back_point))

    def draw(self, action_reward=0.0, action=0):
        """Draw the screen"""
        self.screen.fill((135, 206, 235))
        self.draw_road()
        self.draw_obstacles()
        self.draw_player()
        AI = self.AI

        # Display score
        font = pygame.font.SysFont("Arial", 36)
        self.screen.blit(font.render(f"Score: {self.score}", True, (0, 0, 0)), (20, 20))

        # Display AI status information
        font_state = pygame.font.SysFont("Arial", 24)
        # action = -1: move left, 0: stay, 1: move right, 2: jump
        action = ['Left', 'None', 'Right', 'Jump'][action + 1]
        text_surface = font_state.render(
            f"{'AI Control' if AI else 'Player Control'} | Action: {action} | Action Reward: {action_reward:.1f}",
            True, (0, 0, 0))

        self.screen.blit(text_surface, (20, 60))

        pygame.display.flip()
        self.clock.tick(60)

    def step(self, action):
        """Execute action and return new state, reward, and whether the game is over"""
        done = False
        score = self.score  # Record cumulative score
        reward = 0.0  # Initialize reward

        for event in pygame.event.get():
            if event.type == pygame.WINDOWFOCUSLOST:  # Continue running when the window loses focus
                pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSGAINED))  # Fake focus event
            if event.type == pygame.QUIT:
                self.close()
                exit()

        distance_x = self.obstacles[0]["x"] - self.player_x
        distance_z = self.obstacles[0]["z"]
        # Interpret action
        if action < 2:
            if action == -1:  # Move left
                self.player_x = max(-self.ROAD_WIDTH // 2 + 20, self.player_x - self.player_lateral_speed)
                if distance_x >= 40:
                    reward = -5  # Penalize for moving away from the obstacle
                elif distance_x <= -40:
                    reward = 5
            elif action == 1:  # Move right
                self.player_x = min(self.ROAD_WIDTH // 2 - 20, self.player_x + self.player_lateral_speed)
                if distance_x <= -40:
                    reward = -5  # Penalize for moving away from the obstacle
                elif distance_x >= 40:
                    reward = 5
            elif action == 0:  # Stay
                if abs(distance_x) <= 40:
                    reward = 5
                else:
                    reward = -5

            # Update platform position
            for obstacle in self.obstacles:
                obstacle["z"] -= 10  # Platform moves towards the player
                if obstacle["z"] < 0:
                    self.spawn_obstacle()  # Generate new obstacle
                    self.obstacles.remove(obstacle)  # Remove obstacles that have gone off-screen
                    self.score = 0
                    done = True
            self.draw(action_reward=reward, action=action)

        elif action == 2:  # Jump
            get_Y_score = False
            jumping = True
            self.player_y_velocity = 14.0  # Reset jump speed
            while jumping:

                # Update obstacle positions
                for obstacle in self.obstacles:
                    obstacle["z"] -= 10  # Obstacles move towards the player

                # Update player vertical position
                self.player_y_velocity -= self.GRAVITY  # Gravity effect
                self.player_y += self.player_y_velocity  # Update player y position

                # Check if score is obtained
                if len(self.obstacles) >= 1:
                    if ((abs(self.player_y - self.obstacles[0]["y"]) <= 40) and (abs(self.player_x - self.obstacles[0]["x"]) <= 40)
                            and (abs(self.obstacles[0]["z"]) <= 1e-6)):  # Detect overlap
                        if not get_Y_score:
                            reward = 100.0  # Base reward
                            reward += 10 * (1 - abs(self.player_y - self.obstacles[0]["y"]) / 40)  # Closer distance, higher reward
                            self.score += 1
                            score = self.score  # Update cumulative score
                            get_Y_score = True

                # Check if fallen to the ground
                if self.player_y <= 0:
                    self.player_y = 0
                    if not get_Y_score:
                        reward = -50.0
                    jumping = False  # End jump

                for obstacle in self.obstacles:
                    if obstacle["z"] < 0:
                        self.spawn_obstacle()  # Generate new obstacle
                        self.obstacles.remove(obstacle)  # Remove obstacles that have gone off-screen
                        if not get_Y_score:
                            self.score = 0
                            done = True
                # Draw the screen, including obstacles, player, background, score records, etc.
                self.draw(action_reward=reward, action=action)

        # Get state
        state = self.get_state()

        return state, reward, done, score

    def get_state(self):
        state = []
        state.extend([
            self.obstacles[0]["x"] - self.player_x,
            self.obstacles[0]["z"]
        ])
        return state

    def close(self):
        pygame.quit()