import os
# HEADLESS モード：表示する場合は False に設定してください。
HEADLESS = False
if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame, sys, math, random
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torch.optim as optim

pygame.init()

# ユーザー入力：総エピソード数、初期パック速度、最大壁衝突回数
try:
    total_episodes = int(input("Enter total number of episodes: "))
except Exception as e:
    print("Input error. Using default total episodes = 100.")
    total_episodes = 100

try:
    initial_speed = float(input("Enter initial puck speed: "))
except Exception as e:
    print("Input error. Using default puck speed = 10.0.")
    initial_speed = 10.0

try:
    max_wall_collisions = int(input("Enter maximum allowed wall collisions per episode: "))
except Exception as e:
    print("Input error. Using default max wall collisions = 20.")
    max_wall_collisions = 20

WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-Agent Hyper Hockey")

# 色の定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
YELLOWGREEN = (154, 205, 50)

clock = pygame.time.Clock()

PADDLE_RADIUS = 30
PUCK_RADIUS = 15

####################################
# 予測関数: パックが target_x に到達する際の予測 y 座標（壁反射を考慮）
####################################
def predict_collision_y(puck, target_x, height):
    if puck.vx > 0 and target_x > puck.x:
        t = (target_x - puck.x) / puck.vx
    elif puck.vx < 0 and target_x < puck.x:
        t = (puck.x - target_x) / abs(puck.vx)
    else:
        return puck.y
    y = puck.y
    vy = puck.vy
    while t > 0:
        if vy > 0:
            time_to_bottom = (height - y) / vy
            if t < time_to_bottom:
                y = y + vy * t
                t = 0
            else:
                y = height
                t -= time_to_bottom
                vy = -vy
        elif vy < 0:
            time_to_top = y / abs(vy)
            if t < time_to_top:
                y = y + vy * t
                t = 0
            else:
                y = 0
                t -= time_to_top
                vy = -vy
        else:
            break
    return y

####################################
# Actor-Criticネットワーク（PPO用）
####################################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

####################################
# PPOAgent クラス（共通）
####################################
class PPOAgent:
    def __init__(self, paddle, name, state_dim, action_dim, hidden_dim, lr, gamma, clip_epsilon, update_epochs):
        self.paddle = paddle
        self.name = name
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    def get_state(self, puck):
        # 状態: [paddle.y/HEIGHT, puck.y/HEIGHT, puck.vy/initial_speed]
        state = torch.tensor([self.paddle.y/HEIGHT, puck.y/HEIGHT, puck.vy/initial_speed], dtype=torch.float32)
        return state.to(self.device)
    def choose_action(self, state):
        logits, value = self.policy(state.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    def evaluate_actions(self, states, actions):
        logits, values = self.policy(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, values.squeeze(-1), dist_entropy
    def update(self, trajectory):
        states = torch.stack([torch.tensor(tr[0], dtype=torch.float32) for tr in trajectory]).to(self.device)
        actions = torch.tensor([tr[1] for tr in trajectory], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([tr[2] for tr in trajectory], dtype=torch.float32).to(self.device)
        rewards = [tr[3] for tr in trajectory]
        values = torch.tensor([tr[4] for tr in trajectory], dtype=torch.float32).to(self.device)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        T = len(trajectory)
        returns = returns / T  # エピソード内の平均報酬
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.update_epochs):
            new_log_probs, new_values, entropy = self.evaluate_actions(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - new_values).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

####################################
# SyncAgent（左側エージェント：パックのy座標に同期して動く場合の例）
####################################
class SyncAgent:
    def __init__(self, paddle, name, tolerance=5):
        self.paddle = paddle
        self.name = name
        self.tolerance = tolerance
    def get_state(self, puck):
        return [self.paddle.y/HEIGHT, puck.y/HEIGHT, puck.vy/initial_speed]
    def choose_action(self, state):
        current_y = self.paddle.y
        target_y = state[1] * HEIGHT
        if current_y < target_y - self.tolerance:
            return 2, 0.0, 0.0  # move down
        elif current_y > target_y + self.tolerance:
            return 0, 0.0, 0.0  # move up
        else:
            return 1, 0.0, 0.0  # stay

####################################
# Pygame 用のパドル、パック、衝突処理（再定義）
####################################
class Paddle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.radius = PADDLE_RADIUS
        self.speed = 5
    def move(self, dy):
        self.y += dy
        if self.y - self.radius < 0:
            self.y = self.radius
        if self.y + self.radius > HEIGHT:
            self.y = HEIGHT - self.radius
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Puck:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.radius = PUCK_RADIUS
        self.speed = speed
        self.max_speed = max(8, self.speed * 1.5)
        self.reset_velocity()
    def reset_velocity(self):
        # 発射時、中央から -45度～45度 のランダムな方向で発射
        angle = random.uniform(-math.pi/4, math.pi/4)
        self.vx = self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)
    def move(self):
        self.x += self.vx
        self.y += self.vy
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)
    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.reset_velocity()

def circle_collision(x1, y1, r1, x2, y2, r2):
    return math.hypot(x1 - x2, y1 - y2) < (r1 + r2)

def handle_collision(puck, paddle):
    dx = puck.x - paddle.x
    dy = puck.y - paddle.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        dist = 0.1
    nx = dx / dist
    ny = dy / dist
    dot = puck.vx * nx + puck.vy * ny
    puck.vx = puck.vx - 2 * dot * nx
    puck.vy = puck.vy - 2 * dot * ny
    puck.vx *= 1.05
    puck.vy *= 1.05
    speed = math.hypot(puck.vx, puck.vy)
    if speed > puck.max_speed:
        factor = puck.max_speed / speed
        puck.vx *= factor
        puck.vy *= factor
    overlap = (puck.radius + paddle.radius) - dist
    puck.x += nx * overlap
    puck.y += ny * overlap

####################################
# エピソード開始時にパドルとパックを中央にリセットする関数
####################################
def reset_environment():
    reset_paddles()
    puck.reset()

def reset_paddles():
    left_paddle.y = HEIGHT // 2
    right_paddle.y = HEIGHT // 2

####################################
# ゲーム開始前：壁衝突回数のカウント
####################################
wall_collision_count = 0

####################################
# ゲーム開始前：重みファイルの選択（番号指定）
####################################
weight_files = [f for f in os.listdir('.') if f.endswith('.pt')]
if weight_files:
    print("Available weight files:")
    for idx, fname in enumerate(weight_files):
        print(f"{idx}: {fname}")
    selection = input("Select weight file number (or press Enter to train from scratch): ").strip()
    if selection != "":
        try:
            index = int(selection)
            if 0 <= index < len(weight_files):
                model_path = weight_files[index]
                print(f"Selected model file: {model_path}")
            else:
                print("Invalid selection. Training from scratch.")
                model_path = None
        except Exception as e:
            print("Invalid input. Training from scratch.")
            model_path = None
    else:
        model_path = None
else:
    model_path = None

####################################
# オブジェクト生成
####################################
left_paddle = Paddle(PADDLE_RADIUS + 10, HEIGHT // 2, RED)
right_paddle = Paddle(WIDTH - PADDLE_RADIUS - 10, HEIGHT // 2, BLUE)
puck = Puck(WIDTH // 2, HEIGHT // 2, initial_speed)

# エージェントの生成：左右のエージェントはそれぞれ異なるハイパーパラメータ
left_agent = PPOAgent(left_paddle, "LeftAgent", state_dim=3, action_dim=3, hidden_dim=64, lr=1e-3, gamma=0.99, clip_epsilon=0.2, update_epochs=4)
right_agent = PPOAgent(right_paddle, "RightAgent", state_dim=3, action_dim=3, hidden_dim=64, lr=5e-4, gamma=0.99, clip_epsilon=0.1, update_epochs=8)

if model_path is not None:
    right_agent.policy.load_state_dict(torch.load(model_path, map_location=right_agent.device))
    print(f"Loaded model weights from {model_path}")

score_left = 0
score_right = 0

episode_count = 0
episodes = []
left_scores_list = []
right_scores_list = []
episode_rewards_left = []
episode_rewards_right = []

ppo_trajectory_left = []
ppo_trajectory_right = []
video_frames = []

best_reward_left = -float('inf')
best_reward_right = -float('inf')

####################################
# メインゲームループ
####################################
running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 状態取得
    left_state = left_agent.get_state(puck)
    right_state = right_agent.get_state(puck)
    left_action, left_log_prob, left_value = left_agent.choose_action(left_state)
    right_action, right_log_prob, right_value = right_agent.choose_action(right_state)

    # パドル移動
    if left_action == 0:
        left_paddle.move(-left_paddle.speed)
    elif left_action == 2:
        left_paddle.move(left_paddle.speed)
    if right_action == 0:
        right_paddle.move(-right_paddle.speed)
    elif right_action == 2:
        right_paddle.move(right_paddle.speed)

    puck.move()

    if puck.y - puck.radius <= 0:
        puck.y = puck.radius
        puck.vy *= -1
        wall_collision_count += 1
    if puck.y + puck.radius >= HEIGHT:
        puck.y = HEIGHT - puck.radius
        puck.vy *= -1
        wall_collision_count += 1

    if wall_collision_count >= max_wall_collisions:
        goal_occurred = True
    else:
        goal_occurred = False

    goal_width = 10
    goal_height = HEIGHT // 3
    left_goal_rect = pygame.Rect(0, (HEIGHT - goal_height) // 2, goal_width, goal_height)
    right_goal_rect = pygame.Rect(WIDTH - goal_width, (HEIGHT - goal_height) // 2, goal_width, goal_height)

    # 予測: パックが各エージェントのパドルの x 座標に到達する際の予測 y 座標
    predicted_y_left = predict_collision_y(puck, left_paddle.x, HEIGHT)
    predicted_y_right = predict_collision_y(puck, right_paddle.x, HEIGHT)
    threshold = 0.1 * HEIGHT
    left_diff = abs(left_paddle.y - predicted_y_left)
    right_diff = abs(right_paddle.y - predicted_y_right)
    left_alignment_reward = 10 * max(0, 1 - left_diff/threshold)
    right_alignment_reward = 10 * max(0, 1 - right_diff/threshold)

    # 方向性ボーナス
    if puck.vy > 0:
        left_direction_bonus = 2 if left_action == 2 else (-2 if left_action == 0 else 0)
        right_direction_bonus = 2 if right_action == 2 else (-2 if right_action == 0 else 0)
    elif puck.vy < 0:
        left_direction_bonus = 2 if left_action == 0 else (-2 if left_action == 2 else 0)
        right_direction_bonus = 2 if right_action == 0 else (-2 if right_action == 2 else 0)
    else:
        left_direction_bonus = 0
        right_direction_bonus = 0

    left_reward = left_alignment_reward + left_direction_bonus
    right_reward = right_alignment_reward + right_direction_bonus

    # ゴール判定
    if puck.x - puck.radius <= 0:
        if left_goal_rect.top <= puck.y <= left_goal_rect.bottom:
            score_right += 1
            left_reward -= 5
            right_reward += 5
            puck.reset()
            goal_occurred = True
        else:
            puck.x = puck.radius
            puck.vx *= -1
    if puck.x + puck.radius >= WIDTH:
        if right_goal_rect.top <= puck.y <= right_goal_rect.bottom:
            score_left += 1
            left_reward += 5
            right_reward -= 5
            puck.reset()
            goal_occurred = True
        else:
            puck.x = WIDTH - puck.radius
            puck.vx *= -1

    if circle_collision(puck.x, puck.y, puck.radius, left_paddle.x, left_paddle.y, left_paddle.radius):
        handle_collision(puck, left_paddle)
        left_reward += 2.0
    if circle_collision(puck.x, puck.y, puck.radius, right_paddle.x, right_paddle.y, right_paddle.radius):
        handle_collision(puck, right_paddle)
        right_reward += 2.0

    ppo_trajectory_left.append((left_state.cpu().numpy(), left_action, left_log_prob, left_reward, left_value))
    ppo_trajectory_right.append((right_state.cpu().numpy(), right_action, right_log_prob, right_reward, right_value))

    frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
    video_frames.append(frame)

    if goal_occurred:
        episode_count += 1
        episodes.append(episode_count)
        left_scores_list.append(score_left)
        right_scores_list.append(score_right)
        T_left = len(ppo_trajectory_left)
        T_right = len(ppo_trajectory_right)
        ep_reward_left = sum([tr[3] for tr in ppo_trajectory_left]) / T_left if T_left > 0 else 0
        ep_reward_right = sum([tr[3] for tr in ppo_trajectory_right]) / T_right if T_right > 0 else 0
        episode_rewards_left.append(ep_reward_left)
        episode_rewards_right.append(ep_reward_right)
        loss_left = left_agent.update(ppo_trajectory_left)
        loss_right = right_agent.update(ppo_trajectory_right)
        ppo_trajectory_left = []
        ppo_trajectory_right = []
        print(f"Episode {episode_count}: Left Score = {score_left}, Right Score = {score_right}, Left Reward = {ep_reward_left:.3f}, Right Reward = {ep_reward_right:.3f}")
        if ep_reward_left > best_reward_left:
            best_reward_left = ep_reward_left
            torch.save(left_agent.policy.state_dict(), "best_model_left.pt")
            print(f"New best model (left) saved with reward {best_reward_left:.3f}")
        if ep_reward_right > best_reward_right:
            best_reward_right = ep_reward_right
            torch.save(right_agent.policy.state_dict(), "best_model_right.pt")
            print(f"New best model (right) saved with reward {best_reward_right:.3f}")
        if episode_count % 10 == 0:
            plt.clf()
            plt.plot(episodes, left_scores_list, label="Left Agent Score", color="red")
            plt.plot(episodes, right_scores_list, label="Right Agent Score", color="blue")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Score")
            plt.title("Scores Every 10 Episodes")
            plt.legend()
            plt.savefig("scores.png")
            plt.figure()
            plt.plot(episodes, episode_rewards_left, label="Left Agent Reward", color="red")
            plt.plot(episodes, episode_rewards_right, label="Right Agent Reward", color="blue")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("Reward Progress Every 10 Episodes")
            plt.legend()
            plt.savefig("rewards.png")
            pl t.close()
            # imageio.mimwrite("gameplay.mp4", video_frames, fps=30)
        video_frames = []
        wall_collision_count = 0
        reset_paddles()
        puck.reset()
        if episode_count >= total_episodes:
            running = False

    if not HEADLESS:
        screen.fill(BLACK)
        pygame.draw.line(screen, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)
        pygame.draw.rect(screen, YELLOWGREEN, left_goal_rect)
        pygame.draw.rect(screen, YELLOWGREEN, right_goal_rect)
        left_paddle.draw(screen)
        right_paddle.draw(screen)
        puck.draw(screen)
        font = pygame.font.SysFont("Arial", 48)
        score_text = font.render(f"{score_left} : {score_right}", True, WHITE)
        screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 10))
        pygame.display.flip()

pygame.quit()
sys.exit()