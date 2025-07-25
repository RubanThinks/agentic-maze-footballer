import pygame, random, numpy as np, heapq
from PIL import Image

WIDTH, HEIGHT = 600, 600
ROWS, COLS = 10, 10
CELL_SIZE = WIDTH // COLS

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# === Maze generation ===
def generate_maze(rows, cols):
    return np.random.choice([0, 1], size=(rows, cols), p=[0.7, 0.3])

class MazeEnv:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.maze = generate_maze(rows, cols)
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self.agent = self.start

    def set_points(self, start, goal):
        self.start, self.goal = start, goal
        self.agent = start

    def reset(self):
        self.agent = self.start
        return self.agent

    def step(self, action):
        r, c = self.agent
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1

        if r < 0 or r >= self.rows or c < 0 or c >= self.cols or self.maze[r, c] == 1:
            return self.agent, -50, False

        self.agent = (r, c)
        if self.agent == self.goal:
            return self.agent, 100, True

        return self.agent, -1, False

    def get_state_idx(self, state):
        return state[0] * self.cols + state[1]

# === BFS Reachability ===
def bfs_reachable(env):
    from collections import deque
    q = deque([env.start])
    visited = set([env.start])
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        r, c = q.popleft()
        if (r, c) == env.goal:
            return True
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.rows and 0 <= nc < env.cols and env.maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return False

# === A* Pathfinding ===
def astar_path(env):
    start, goal = env.start, env.goal
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)
        came_from[current] = parent
        if current == goal:
            path = []
            cur = current
            while cur:
                path.append(cur)
                cur = came_from[cur]
            return list(reversed(path))

        for dr, dc in dirs:
            nr, nc = current[0] + dr, current[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < env.rows and 0 <= nc < env.cols):
                continue
            if env.maze[nr, nc] == 1:
                continue
            tentative_g = cost + 1
            if tentative_g < g_score.get(nxt, 9999):
                g_score[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_set, (f, tentative_g, nxt, current))
    return []

# === RL Training ===
def train_agent_live(env, screen, episodes=10):
    n_states = env.rows * env.cols
    n_actions = 4
    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(150):
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return Q

            s_idx = env.get_state_idx(state)
            action = random.randint(0, n_actions - 1) if random.random() < epsilon else np.argmax(Q[s_idx])

            next_state, reward, done = env.step(action)
            total_reward += reward
            ns_idx = env.get_state_idx(next_state)

            Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action])
            state = next_state
            if done:
                break

            draw_maze(screen, env)
            pygame.display.flip()

        print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward}")

    print("✅ Training finished")
    return Q

def get_rl_optimal_path(env, Q):
    path = []
    visited = set()
    state = env.start
    done = False
    steps = 0
    while not done and steps < 500:
        if state in visited:
            break
        visited.add(state)
        path.append(state)
        s_idx = env.get_state_idx(state)
        action = np.argmax(Q[s_idx])
        next_state, _, done = env.step(action)
        if env.maze[next_state[0], next_state[1]] == 1:
            break
        if abs(next_state[0] - state[0]) + abs(next_state[1] - state[1]) != 1:
            break
        state = next_state
        steps += 1
    if state == env.goal:
        path.append(env.goal)
    return path

# === GIF/Frame Loader ===
def load_gif_frames(path, size):
    pil_gif = Image.open(path)
    frames = []
    try:
        while True:
            frame = pil_gif.copy().convert("RGBA")
            frame = frame.resize(size, Image.Resampling.LANCZOS)
            pygame_frame = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames.append(pygame_frame)
            pil_gif.seek(pil_gif.tell() + 1)
    except EOFError:
        pass
    return frames

# === Draw Maze ===
def draw_maze(screen, env):
    for r in range(env.rows):
        for c in range(env.cols):
            screen.blit(grass_img_scaled, (c * CELL_SIZE, r * CELL_SIZE))
            if env.maze[r, c] == 1:
                wall_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                wall_surface.fill((0, 0, 0, 150))
                screen.blit(wall_surface, (c * CELL_SIZE, r * CELL_SIZE))

    sr, sc = env.start
    screen.blit(football_img_scaled, (sc * CELL_SIZE, sr * CELL_SIZE))

    gr, gc = env.goal
    screen.blit(goalpost_img_scaled, (gc * CELL_SIZE, gr * CELL_SIZE))

    pygame.display.flip()

# === Victory Celebration ===
def show_victory_animation(screen):
    start_time = pygame.time.get_ticks()
    duration = 3000  # 3 seconds

    # Play celebration GIF for 3 seconds
    while pygame.time.get_ticks() - start_time < duration:
        frame = celebration_frames[(pygame.time.get_ticks() // 80) % len(celebration_frames)]
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        pygame.time.delay(50)

    # Then show VICTORY text
    screen.fill((0, 0, 0))
    font = pygame.font.SysFont("Arial", 60, bold=True)
    text = font.render(" GOAL ! ", True, (255, 215, 0))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.delay(2000)  # show for 2 seconds

# === Smooth Rolling ===
def animate_smooth_path(screen, env, path):
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        x1, y1 = c1 * CELL_SIZE, r1 * CELL_SIZE
        x2, y2 = c2 * CELL_SIZE, r2 * CELL_SIZE

        for step in range(12):
            t = step / 12
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t

            draw_maze(screen, env)
            frame = rolling_frames[pygame.time.get_ticks() // 50 % len(rolling_frames)]
            screen.blit(frame, (x, y))
            pygame.display.flip()
            pygame.time.delay(30)

    # ✅ After reaching the goal, show celebration
    show_victory_animation(screen)

# === Solve Maze ===
def solve_maze(env, screen):
    if not bfs_reachable(env):
        print("Goal unreachable -> using A*")
        return astar_path(env)

    Q = train_agent_live(env, screen, episodes=8)
    rl_path = get_rl_optimal_path(env, Q)

    if not rl_path or rl_path[-1] != env.goal:
        print("RL failed, fallback to A*")
        return astar_path(env)
    return rl_path

# === MAIN ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Football Maze Agent")

# Load images
grass_img = pygame.image.load("grass_field.png")
football_img = pygame.image.load("football.png")
goalpost_img = pygame.image.load("goal_post.png")

grass_img_scaled = pygame.transform.scale(grass_img, (CELL_SIZE, CELL_SIZE))
football_img_scaled = pygame.transform.scale(football_img, (CELL_SIZE, CELL_SIZE))
goalpost_img_scaled = pygame.transform.scale(goalpost_img, (CELL_SIZE, CELL_SIZE))

# Load GIFs
rolling_frames = load_gif_frames("football.gif", (CELL_SIZE, CELL_SIZE))
celebration_frames = load_gif_frames("celebration.gif", (WIDTH, HEIGHT))  # Full screen

env = MazeEnv(ROWS, COLS)
clicks = []
final_path = None
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and len(clicks) < 2:
            x, y = event.pos
            r, c = y // CELL_SIZE, x // CELL_SIZE
            clicks.append((r, c))
            if len(clicks) == 2:
                env.set_points(clicks[0], clicks[1])
                final_path = solve_maze(env, screen)
                animate_smooth_path(screen, env, final_path)
                running = False  
    draw_maze(screen, env)

pygame.quit()
