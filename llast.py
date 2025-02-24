import numpy as np
import pygame
import random
import math

from pygame import gfxdraw

class GameEnvironment:
    def __init__(self,AI=True):
        self.player_radius = 20
        self.player_z = 0
        self.ROAD_DEPTH = 4000
        self.player_lateral_speed = 20
        self.ROAD_WIDTH = 1000
        self.OBSTACLE_HEIGHT = 50
        self.OBSTACLE_HEIGHT_large = 300
        self.OBSTACLE_LENGTH = 40
        self.OBSTACLE_WIDTH = 40
        self.GRAVITY = 0.7
        self.AI = AI
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF  # Enable hardware acceleration and double buffering
        )
        pygame.display.set_caption("Just Jump! - AI Control" if AI else "Just Jump! - Player Control")
        self.clock = pygame.time.Clock()

        self.reset()
        self.draw()

    def reset(self):
        self.player_x = 0
        self.player_y = 0
        self.player_velocity_y = 14.0
        #self.game_over = False
        self.score = 0
        self.obstacles = []
        # 生成初始障碍物
        self.spawn_obstacle()

        return self.get_state()

    def spawn_obstacle(self):
        # Generate first obstacle if none exists
        if len(self.obstacles) == 0:
            is_small_obstacle = random.random() < 0.7  # 70 per cent probability of generating small obstacles
            self.obstacles.append({
                "x": random.randrange(-self.ROAD_WIDTH // 2 + self.OBSTACLE_WIDTH // 2 ,-self.ROAD_WIDTH // 2 + self.OBSTACLE_WIDTH // 2 + 60,20),
                "y": 0,
                "z": self.ROAD_DEPTH - 600,
                "height": self.OBSTACLE_HEIGHT if is_small_obstacle else self.OBSTACLE_HEIGHT_large
            })

        # ==== Z-axis generation rules ====
        # Get the position of the tail of the previous obstacle (smaller z-coordinate means closer to the player)
        last_obstacle = self.obstacles[-1]
        last_z = last_obstacle["z"]
        last_x = last_obstacle["x"]

        # Minimum permissible position of head of new obstacle = tail of previous obstacle + minimum spacing
        min_z = min(last_z + 500, self.ROAD_DEPTH - 100)
        # Maximum permissible position of the head of the new obstacle = tail of the previous obstacle + maximum spacing
        max_z = min(last_z + 1500, self.ROAD_DEPTH - 100)

        new_z = random.randrange(min_z, max_z + 1, 10)
        # ==== X-axis generation rules ====
        min_x = max(-self.ROAD_WIDTH // 2 + self.OBSTACLE_WIDTH // 2 , last_x - 300)
        max_x = min(self.ROAD_WIDTH // 2 - self.OBSTACLE_WIDTH // 2 , last_x + 300)

        # Excluding the intermediate 150 distance range
        left_min = min_x
        left_max = last_x - 100
        right_min = last_x + 100
        right_max = max_x

        possible_ranges = []
        if left_min <= left_max:
            possible_ranges.append((left_min, left_max))
        if right_min <= right_max:
            possible_ranges.append((right_min, right_max))

        if possible_ranges:
            # Randomly select a valid interval and generate the x-coordinate
            selected_min, selected_max = random.choice(possible_ranges)
            new_x = random.randrange(selected_min, selected_max + 1, 20)
        else:
            # If the valid interval cannot be found, fall back to the original way of selecting the x-coordinate
            new_x = random.randrange(min_x, max_x + 1, 20)


        # ==== Generating new obstacles ====
        is_small_obstacle = random.random() < 0.7  # 70 per cent probability of generating small obstacles
        new_obstacle = {
            "x": new_x,
            "y": 0,  # Height of the base of the obstacle
            "z": new_z,  # Initial position of obstacle head
            "height": self.OBSTACLE_HEIGHT if is_small_obstacle else self.OBSTACLE_HEIGHT_large
        }
        self.obstacles.append(new_obstacle)

    def project_3d_to_2d(self, x, y, z):
        """Projection function with top view effect"""
        rel_z = z - (self.player_z - 500)  # Correct rel_z calculations to account for camera distances
        min_z = 1
        effective_z = max(rel_z, min_z)

        fov = math.radians(60)
        scale = self.WIDTH / (2 * math.tan(fov))

        # Add the top view offset
        y_offset = 200 + math.tan(math.radians(30)) * rel_z

        screen_x = (x * scale / effective_z) + self.WIDTH // 2
        screen_y = self.HEIGHT // 2 - ((y - y_offset) * scale / effective_z)  # Subtract to invert the y-axis

        return screen_x, screen_y

    def draw_player(self):
        """Draw the player ball with adjusted transparency"""
        player_screen_pos = self.project_3d_to_2d(0, self.player_y, self.player_radius) #实际上Z=0
        x, y = int(player_screen_pos[0]), int(player_screen_pos[1])

        # Ground projection parameters (varies with height)
        shadow_alpha = max(100 - int(self.player_y * 0.7), 0)
        shadow_scale = 1 - self.player_y / 200
        shadow_width = int(self.player_radius * 1.2 * shadow_scale)
        shadow_height = int(self.player_radius * 0.6 * shadow_scale)

        # Drawing ground projections (ellipses)
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
        """Dynamic ground generation (covering the entire depth of the road)"""
        segment_depth = 50
        start_z = self.player_z - 500  # Generate from camera position
        end_z = start_z + self.ROAD_DEPTH  # End position
        num_segments = int((end_z - start_z) / segment_depth) + 1  # Calculate total number of paragraphs

        for i in range(num_segments):
            z_start = start_z + i * segment_depth
            z_end = z_start + segment_depth

            # Ground Vertex Projection
            left_start = self.project_3d_to_2d(-self.ROAD_WIDTH / 2 - self.player_x, 0, z_start)
            right_start = self.project_3d_to_2d(self.ROAD_WIDTH / 2 - self.player_x, 0, z_start)
            left_end = self.project_3d_to_2d(-self.ROAD_WIDTH / 2 - self.player_x, 0, z_end)
            right_end = self.project_3d_to_2d(self.ROAD_WIDTH / 2 - self.player_x, 0, z_end)

            # Filter segments behind the camera
            if z_end <= self.player_z - 500:
                continue

            # Mapping the ground
            pygame.draw.polygon(self.screen, (192, 192, 192), [
                (left_start[0], left_start[1]),
                (right_start[0], right_start[1]),
                (right_end[0], right_end[1]),
                (left_end[0], left_end[1])
            ], 0)

            # Drawing road margins
            pygame.draw.line(self.screen, (0, 0, 0), left_start, left_end, 2)
            pygame.draw.line(self.screen, (0, 0, 0), right_start, right_end, 2)

        # Draw the disappearing line of the obstacle (red horizontal line)
        left_disappear = self.project_3d_to_2d(-self.ROAD_WIDTH / 2 - self.player_x, 0, -60)
        right_disappear = self.project_3d_to_2d(self.ROAD_WIDTH / 2 - self.player_x, 0, -60)

        # Drawing red horizontal lines
        pygame.draw.line(self.screen, (255, 0, 0), left_disappear, right_disappear, 2)

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            x = obstacle["x"] - self.player_x  # Calculate lateral relative position
            z = obstacle["z"]
            y = obstacle["y"]
            obstacle_length = self.OBSTACLE_LENGTH

            # Vertex y-coordinate corrected to positive
            front_tl = self.project_3d_to_2d(x - self.OBSTACLE_WIDTH / 2, obstacle["height"] + y, z)
            front_tr = self.project_3d_to_2d(x + self.OBSTACLE_WIDTH / 2, obstacle["height"] + y, z)
            front_bl = self.project_3d_to_2d(x - self.OBSTACLE_WIDTH / 2, obstacle["y"], z)
            front_br = self.project_3d_to_2d(x + self.OBSTACLE_WIDTH / 2, obstacle["y"], z)

            back_tl = self.project_3d_to_2d(x - self.OBSTACLE_WIDTH / 2, obstacle["height"] + y, z + obstacle_length)
            back_tr = self.project_3d_to_2d(x + self.OBSTACLE_WIDTH / 2, obstacle["height"] + y, z + obstacle_length)
            back_bl = self.project_3d_to_2d(x - self.OBSTACLE_WIDTH / 2, obstacle["y"], z + obstacle_length)
            back_br = self.project_3d_to_2d(x + self.OBSTACLE_WIDTH / 2, obstacle["y"], z + obstacle_length)

            # Drawing the faces
            surfaces = [
                ([front_tl, front_tr, front_br, front_bl], (100, 100, 100)),  # Front surface
                ([front_tl, back_tl, back_tr, front_tr], (80, 80, 80)),  # Top surface
                ([front_tr, front_br, back_br, back_tr], (60, 60, 60)),  # Right surface
                ([back_tl, back_bl, back_br, back_tr], (40, 40, 40)),  # Rear surface
                ([front_bl, back_bl, back_br, front_br], (20, 20, 20))  # Bottom surface
            ]

            for points, color in surfaces:
                pygame.draw.polygon(self.screen, color, [(p[0], p[1]) for p in points])

    def draw(self, action_reward=0.0, action=0):
        """绘制画面"""
        self.screen.fill((135, 206, 235))
        self.draw_road()
        self.draw_obstacles()
        self.draw_player()
        AI=self.AI
        # 显示分数
        font = pygame.font.SysFont("Arial", 36)
        self.screen.blit(font.render(f"Score: {self.score}", True, (0, 0, 0)), (20, 20))

        # 显示AI状态信息
        font_state = pygame.font.SysFont("Arial", 24)
        action = ["Left", "Right", "Jump", "None"][action]
        text_surface = font_state.render(
            f"{'AI Control' if AI else 'Player Control'} | Action: {action} | Action Reward: {action_reward:.1f}",
            True, (0, 0, 0))
        self.screen.blit(text_surface, (20, 60))

        pygame.display.flip()
        self.clock.tick(60)

    def step(self, action):
        """Perform actions and return new states, rewards, and whether the game is over"""
        done = False
        score = self.score  # Record cumulative scores
        reward = 0.0  # Initialisation incentives

        for event in pygame.event.get():
            if event.type == pygame.WINDOWFOCUSLOST:  # Continue to run when the window loses focus
                pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSGAINED))  # Fake focus events
            if event.type == pygame.QUIT:
                self.close()
                exit()

        distance_x = self.obstacles[0]["x"] - self.player_x
        # Explain the action
        if action < 2:
            right = 1 if self.obstacles[0]["height"] == self.OBSTACLE_HEIGHT else -1
            # Update obstacle locations
            for obstacle in self.obstacles:
                obstacle["z"] -= 15  # Obstacles approaching the player

            if action == -1:  # Left shift
                self.player_x = max(-self.ROAD_WIDTH // 2 + 20, self.player_x - self.player_lateral_speed)
                if right == 1:
                    if distance_x >= 40:
                        reward = -10   # Small obstacle at or to the right of the boundary of the scoring area to the right of the wicket, wicket moves to the left (away from and out of the scoring area), penalty is continually awarded
                    elif distance_x <= -40:
                        reward = 10  # Small obstacle at or to the left of the boundary of the scoring area to the left of the ball, the ball moves to the left (out of the scoring area or close to the scoring area), awarding a prize

                else:
                    if -40< distance_x < 0:
                        reward = -5   # Large obstacle at or to the left of the boundary of the scoring area to the left of the wicket, wicket moves to the left (into or on the demerit area), penalty continuously awarded
                    elif 40 > distance_x >= 0:
                        reward = 5 # Large obstacle at or to the right of the boundary of the scoring area to the right of the wicket, wicket moves to the left (out of the scoring area or away from the scoring area) to give a bonus
            elif action == 1:  # 右移
                self.player_x = min(self.ROAD_WIDTH // 2 - 20, self.player_x + self.player_lateral_speed)
                if right == 1:
                    if distance_x <= -40:
                        reward = -10  # Small obstacle at or to the left of the boundary of the scoring area to the left of the ball, the ball moves to the right (out of the scoring area) and a penalty is awarded
                    elif distance_x >= 40:
                        reward = 10  # Small obstacle at or to the right of the boundary of the scoring area to the right of the ball, the ball moves to the right (close to the scoring area) and a bonus is awarded
                else:
                    if  40 > distance_x > 0:
                        reward = -5
                    elif -40< distance_x <= 0:
                        reward = 5
            elif action == 0:  # The horizontal coordinates don't move
                if abs(distance_x) <= 40:
                    reward = 10 * right # The ball is in the scoring area, giving a bonus
                else:
                    reward = -10 * right # The wicket is not in the scoring area and a penalty is awarded

            if self.check_done():
                done = True
                reward -= 50.0
                self.score = 0
                self.obstacles.remove(self.obstacles[0])  # Remove obstacles that have gone beyond the screen

            # 更新障碍物
            if len(self.obstacles) <= 8:
                if self.obstacles[-1]["z"] < self.ROAD_DEPTH - 600:
                    self.spawn_obstacle()
            if self.obstacles[0]["z"] < (
                    -(self.player_radius + self.OBSTACLE_LENGTH)):  # and len(self.obstacles) > 1
                self.obstacles.remove(self.obstacles[0])  # Remove obstacles that have gone beyond the screen

            self.draw(action_reward=reward, action=action)
        if action == 2:  # Jumping
            jumping = True
            self.player_y_velocity = 14.0  # Reset jump speed
            get_score = False
            while jumping:

                # Update obstacle locations
                for obstacle in self.obstacles:
                    obstacle["z"] -= 15  # Obstacles approaching the player

                # Update player vertical position
                self.player_y_velocity -= self.GRAVITY  # Gravitational action
                self.player_y += self.player_y_velocity  # Update the player's position y

                # Determine if a score was scored
                if len(self.obstacles) >= 1:
                    if ((self.player_y - self.obstacles[0]["height"] > 0)
                            and (abs(self.player_x - self.obstacles[0]["x"]) <= (self.OBSTACLE_WIDTH // 2+ self.player_radius))
                            and (-(self.player_radius + self.OBSTACLE_LENGTH) <= self.obstacles[0]["z"] <= self.player_radius)):# 检测到重叠
                        if not get_score:
                            get_score = True


                # If you hit an obstacle in the middle of a jump
                if self.check_done():
                    done = True

                # Check for falling to the ground
                if self.player_y <= 0:
                    self.player_y = 0
                    if get_score and not done:
                        reward = 100.0  # Base incentives
                        self.score += 1
                        score = self.score  # Update cumulative scores
                    if not get_score:
                        reward = -50.0  # Penalties for meaningless jumps
                    if done:
                        reward -= 50.0
                        self.score = 0
                        self.obstacles.remove(self.obstacles[0])  # Remove colliding obstacles
                    jumping = False  # End of the jump

                if len(self.obstacles) <= 8:
                    if self.obstacles[-1]["z"] < self.ROAD_DEPTH - 600:
                        self.spawn_obstacle()

                if self.obstacles[0]["z"] < (
                        -(self.player_radius + self.OBSTACLE_LENGTH)):
                    self.obstacles.remove(self.obstacles[0])  # Remove obstacles that have gone beyond the screen

                # Drawing screens, including obstacles, players, backgrounds, records, etc.
                self.draw(action_reward=reward, action=action)

        # Get status
        state = self.get_state()

        return state, reward,done, score

    def get_state(self):
        state = []
        state.extend([
                      self.obstacles[0]["x"] - self.player_x,
                      self.obstacles[0]["z"],
                      1 if self.obstacles[0]["height"] == self.OBSTACLE_HEIGHT else -1,
                     ])
        return state

    def check_done(self):# Check for collisions
        for obstacle in self.obstacles:
            distance_x = obstacle["x"] - self.player_x
            distance_z = obstacle["z"]
            distance_y = self.player_y - obstacle["height"]
            if ((abs(distance_x) <= (self.OBSTACLE_WIDTH // 2 + self.player_radius))
                    and ((-(self.player_radius + self.OBSTACLE_LENGTH)) <= distance_z <= self.player_radius)
                    and (distance_y <= 0)):
                return True
        return False

    def close(self):
        pygame.quit()