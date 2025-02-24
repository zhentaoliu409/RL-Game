import pygame
import sys
import importlib
import multiprocessing as mp
import threading
from time import sleep

# Colour and layout configuration
COLORS = {
    "background": (30, 30, 30),
    "button": (70, 130, 180),
    "text": (255, 255, 255),
    "hover": (100, 150, 200)
}

class MainMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Menu")
        self.buttons = [
            {"rect": pygame.Rect(100, 100, 220, 50), "text": "Mode 1 Pluck Stars", "action": 1},
            {"rect": pygame.Rect(100, 180, 220, 50), "text": "Mode 2 Just Jump!", "action": 2}
        ]
        self.font = pygame.font.Font(None, 36)

    def draw_buttons(self):
        self.screen.fill(COLORS["background"])
        for btn in self.buttons:
            rect = btn["rect"]
            color = COLORS["hover"] if rect.collidepoint(pygame.mouse.get_pos()) else COLORS["button"]
            pygame.draw.rect(self.screen, color, rect)
            text = self.font.render(btn["text"], True, COLORS["text"])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def run(self):
        while True:
            self.draw_buttons()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for btn in self.buttons:
                        if btn["rect"].collidepoint(pos):
                            pygame.quit()
                            return btn["action"]
            pygame.display.flip()


def run_player_process(mode, exit_event):
    """Player-controlled processes"""
    prefix = "last" if mode == 1 else "llast"
    GameEnv = importlib.import_module(prefix).GameEnvironment
    env = GameEnv(AI=False)

    # Input state tracker
    input_state = {
        pygame.K_LEFT: {"pressed": False, "last_time": 0},
        pygame.K_RIGHT: {"pressed": False, "last_time": 0}
    }

    # Time parameters (in milliseconds)
    INITIAL_DELAY = 80  # First response delay
    REPEAT_INTERVAL = 5  # Continuous response interval

    while not exit_event.is_set():
        current_time = pygame.time.get_ticks()
        action = 0
        keys = pygame.key.get_pressed()
        # Detect changes in arrow key status
        for key in [pygame.K_LEFT, pygame.K_RIGHT]:
            if keys[key]:
                if not input_state[key]["pressed"]:  # Press for the first time
                    input_state[key]["pressed"] = True
                    input_state[key]["last_time"] = current_time
                    action = -1 if key == pygame.K_LEFT else 1
                else:
                    elapsed = current_time - input_state[key]["last_time"]
                    if elapsed > INITIAL_DELAY:
                        if elapsed % REPEAT_INTERVAL < 5:
                            action = -1 if key == pygame.K_LEFT else 1
            else:
                input_state[key]["pressed"] = False

        for event in pygame.event.get():# exit event and space key event
            if event.type == pygame.QUIT:
                exit_event.set()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 2

        # Update game status
        _, _, done, _ = env.step(action)



def run_ai_process(mode, exit_event):
    """AI Demonstration Process"""
    prefix = "last" if mode == 1 else "llast"
    GameEnv = importlib.import_module(prefix).GameEnvironment
    QLearningAgent = importlib.import_module(f"{prefix}_ai").QLearningAgent


    env = GameEnv()
    agent = QLearningAgent(env, epsilon=0)
    agent.load_model()

    state = env.get_state()
    while not exit_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_event.set()

        # AI决策
        action = agent.get_action(state)
        next_state, _, done, _ = env.step(action)
        state = next_state

def process_monitor(processes, exit_event):
    """Process Monitor Threads"""
    while not exit_event.is_set():
        for p in processes:
            if not p.is_alive():
                exit_event.set()
        sleep(0.1)


if __name__ == "__main__":
    # Display main menu and get mode selection
    while True:
        menu = MainMenu()
        selected_mode = menu.run()
        exit_event = mp.Event()
        # Create and start processes
        processes = [
            mp.Process(target=run_player_process, args=(selected_mode, exit_event)),
            mp.Process(target=run_ai_process, args=(selected_mode, exit_event))
        ]

        for p in processes:
            p.start()

        monitor = threading.Thread(target=process_monitor, args=(processes, exit_event))
        monitor.daemon = True
        monitor.start()

        # Wait for the process to exit
        while any(p.is_alive() for p in processes):
            sleep(0.1)

        # Cleaning up the process
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()
