import pygame, random, numpy as np, heapq

WIDTH, HEIGHT = 600, 600
ROWS, COLS = 10, 10
CELL_SIZE = WIDTH // COLS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255,165,0)

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def generate_maze(rows, cols):
    maze = np.random.choice([0, 1], size=(rows, cols), p=[0.7, 0.3])
    return maze

class MazeEnv:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.maze = generate_maze(rows, cols)
        self.start = (0,0)
        self.goal = (rows-1, cols-1)
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

        if r < 0 or r >= self.rows or c < 0 or c >= self.cols or self.maze[r,c] == 1:
            return self.agent, -50, False

        self.agent = (r,c)
        if self.agent == self.goal:
            return self.agent, 100, True

        return self.agent, -1, False

    def get_state_idx(self, state):
        return state[0]*self.cols + state[1]

def bfs_reachable(env):
    """Check if goal reachable from start"""
    from collections import deque
    q = deque([env.start])
    visited = set([env.start])
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        r,c = q.popleft()
        if (r,c)==env.goal:
            return True
        for dr,dc in dirs:
            nr,nc = r+dr, c+dc
            if 0<=nr<env.rows and 0<=nc<env.cols and env.maze[nr,nc]==0 and (nr,nc) not in visited:
                visited.add((nr,nc))
                q.append((nr,nc))
    return False

def astar_path(env):
    """A* pathfinding as fallback"""
    start, goal = env.start, env.goal
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    def heuristic(a,b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])

    open_set=[]
    heapq.heappush(open_set,(0+heuristic(start,goal),0,start,None))
    came_from={}
    g_score={start:0}

    while open_set:
        _,cost,current,parent=heapq.heappop(open_set)
        came_from[current]=parent
        if current==goal:
            path=[]
            cur=current
            while cur:
                path.append(cur)
                cur=came_from[cur]
            return list(reversed(path))

        for dr,dc in dirs:
            nr,nc=current[0]+dr,current[1]+dc
            nxt=(nr,nc)
            if not (0<=nr<env.rows and 0<=nc<env.cols):
                continue
            if env.maze[nr,nc]==1:
                continue
            tentative_g=cost+1
            if tentative_g<g_score.get(nxt,9999):
                g_score[nxt]=tentative_g
                f=tentative_g+heuristic(nxt,goal)
                heapq.heappush(open_set,(f,tentative_g,nxt,current))
    return []

def train_agent_live(env, screen, episodes=200):
    n_states = env.rows * env.cols
    n_actions = 4
    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            s_idx = env.get_state_idx(state)

            if random.random() < epsilon:
                action = random.randint(0, n_actions-1)
            else:
                action = np.argmax(Q[s_idx])

            next_state, reward, done = env.step(action)
            ns_idx = env.get_state_idx(next_state)
            Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action])
            state = next_state
            steps += 1

            print(f"Episode {ep+1}/{episodes} | Step {steps} | State {state} | Reward {reward}")

            draw_maze(screen, env)
            ar, ac = env.agent
            pygame.draw.rect(screen, ORANGE, (ac*CELL_SIZE, ar*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.display.flip()
            pygame.time.delay(30)
            pygame.event.pump()

    return Q

def get_rl_optimal_path(env, Q):
    path = []
    visited=set()
    state = env.start
    done=False
    steps=0
    while not done and steps<500:
        if state in visited: break
        visited.add(state)
        path.append(state)
        s_idx=env.get_state_idx(state)
        action=np.argmax(Q[s_idx])
        next_state,_,done=env.step(action)
        if env.maze[next_state[0],next_state[1]]==1: break
        if abs(next_state[0]-state[0])+abs(next_state[1]-state[1])!=1: break
        state=next_state
        steps+=1
    if state==env.goal: path.append(env.goal)
    return path

def draw_maze(screen, env, path=None):
    screen.fill(WHITE)
    for r in range(env.rows):
        for c in range(env.cols):
            rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if env.maze[r,c]==1:
                pygame.draw.rect(screen, BLACK, rect)
            pygame.draw.rect(screen,(200,200,200),rect,1)

    sr,sc=env.start
    gr,gc=env.goal
    pygame.draw.rect(screen,GREEN,(sc*CELL_SIZE,sr*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    pygame.draw.rect(screen,RED,(gc*CELL_SIZE,gr*CELL_SIZE,CELL_SIZE,CELL_SIZE))

    if path and len(path)>1:
        pts=[(c*CELL_SIZE+CELL_SIZE//2,r*CELL_SIZE+CELL_SIZE//2) for (r,c) in path]
        pygame.draw.lines(screen,BLUE,False,pts,5)
    pygame.display.flip()

def animate_path(screen, env, path):
    for (r,c) in path:
        draw_maze(screen, env, path[:path.index((r,c))+1])
        pygame.display.flip()
        pygame.time.delay(150)

def solve_maze(env,screen):
    if not bfs_reachable(env):
        print("Goal unreachable -> using A*")
        return astar_path(env)

    Q=train_agent_live(env,screen,episodes=20)
    rl_path=get_rl_optimal_path(env,Q)

    if not rl_path or rl_path[-1]!=env.goal:
        print("RL failed, fallback to A*")
        return astar_path(env)
    return rl_path

pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("RL Maze Agent + A* Fallback")

env=MazeEnv(ROWS,COLS)
clicks=[]
training_done=False
final_path=None
running=True

while running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
        if event.type==pygame.MOUSEBUTTONDOWN and len(clicks)<2:
            x,y=event.pos
            r,c=y//CELL_SIZE,x//CELL_SIZE
            clicks.append((r,c))
            if len(clicks)==2:
                env.set_points(clicks[0],clicks[1])
                final_path=solve_maze(env,screen)
                training_done=True
                animate_path(screen,env,final_path)
    draw_maze(screen,env,final_path if training_done else None)
pygame.quit()