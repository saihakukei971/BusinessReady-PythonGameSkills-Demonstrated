#!/usr/bin/env python
"""
Puyo Puyo (Cute Version with Custom Board Size and Scaling)
※このコードは、Python の仮想環境内で動作させることを想定しています。

【仮想環境の構築手順（例：Windowsの場合）】
1. デスクトップ上に "puyopuyo" フォルダを作成する。
2. コマンドプロンプト（またはターミナル）を開き、以下のようにフォルダに移動する:
     cd %USERPROFILE%\Desktop\puyopuyo
   （Mac/Linuxの場合は "cd ~/Desktop/puyopuyo"）
3. 仮想環境を作成する:
     python -m venv venv
4. 仮想環境を有効化する:
     Windows: venv\Scripts\activate
     Mac/Linux: source venv/bin/activate
5. Pygame をインストールする:
     pip install pygame
6. このスクリプト（puyopuyo.py）を実行する:
     python puyopuyo.py
    
【操作方法】（起動時に画面中央に表示）
  ←   : Move Left  
  →   : Move Right  
  ↓   : Fast Drop  
  Z    : Rotate Left  
  X    : Rotate Right  
"""

import pygame
import sys
import random
import math

# --- 盤面サイズ入力 ---
def get_board_size():
    try:
        row_input = input("Enter board rows (default 12): ")
        col_input = input("Enter board columns (default 6): ")
        rows = int(row_input) if row_input.strip() != "" else 12
        cols = int(col_input) if col_input.strip() != "" else 6
        return rows, cols
    except Exception as e:
        return 12, 6

# --- 基本の定数（縮尺前） ---
BASE_CELL_SIZE = 40    # 基準セルサイズ
EXTRA_X = 200          # 右側の情報表示領域幅
EXTRA_Y = 40           # 画面上下の余白

# FPS, 落下速度、色などはそのまま
FPS = 120
NORMAL_FALL_SPEED = 100    # 通常時落下速度 (ピクセル/秒)
FAST_FALL_SPEED   = 150    # 高速落下速度 (ピクセル/秒)
BLACK   = (  0,   0,   0)
GRAY    = (100, 100, 100)
WHITE   = (255, 255, 255)
RED     = (255,   0,   0)
GREEN   = (  0, 255,   0)
BLUE    = (  0,   0, 255)
YELLOW  = (255, 255,   0)
PUYO_COLORS = [RED, GREEN, BLUE, YELLOW]
OFFSET_X = 20
OFFSET_Y = 20

# --- ぷよ描画用の補助関数 ---
def draw_cute_puyo(surface, center, radius, color):
    """
    center: (x, y) 座標
    radius: ぷよの半径
    color: ぷよの基本色 (RGBタプル)
    
    ぷよ本体の円、暗めのアウトライン、白い目（黒い瞳付き）、笑顔の弧を描画します。
    """
    pygame.draw.circle(surface, color, center, radius)
    darker = (max(color[0]-50, 0), max(color[1]-50, 0), max(color[2]-50, 0))
    pygame.draw.circle(surface, darker, center, radius, 2)
    eye_radius = max(2, radius // 6)
    eye_offset_x = radius // 3
    eye_offset_y = radius // 3
    left_eye_center = (center[0] - eye_offset_x, center[1] - eye_offset_y)
    right_eye_center = (center[0] + eye_offset_x, center[1] - eye_offset_y)
    pygame.draw.circle(surface, WHITE, left_eye_center, eye_radius)
    pygame.draw.circle(surface, WHITE, right_eye_center, eye_radius)
    pupil_radius = max(1, eye_radius // 2)
    pygame.draw.circle(surface, BLACK, left_eye_center, pupil_radius)
    pygame.draw.circle(surface, BLACK, right_eye_center, pupil_radius)
    smile_rect = pygame.Rect(0, 0, radius, radius//2)
    smile_rect.center = (center[0], center[1] + radius//8)
    pygame.draw.arc(surface, BLACK, smile_rect, math.radians(20), math.radians(160), 2)

# --- クラス定義 ---
class Board:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
    
    def is_inside(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows

    def can_move(self, x, y):
        if not (0 <= x < self.cols):
            return False
        if y >= self.rows:
            return False
        if y < 0:
            return True
        return self.grid[y][x] is None

    def add_puyo(self, x, y, color):
        if self.is_inside(x, y):
            self.grid[y][x] = color

    def remove_puyo(self, x, y):
        if self.is_inside(x, y):
            self.grid[y][x] = None

    def find_groups(self):
        visited = [[False] * self.cols for _ in range(self.rows)]
        groups = []
        for y in range(self.rows):
            for x in range(self.cols):
                if self.grid[y][x] is not None and not visited[y][x]:
                    color = self.grid[y][x]
                    group = []
                    stack = [(x, y)]
                    visited[y][x] = True
                    while stack:
                        cx, cy = stack.pop()
                        group.append((cx, cy))
                        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if self.is_inside(nx, ny) and not visited[ny][nx] and self.grid[ny][nx] == color:
                                visited[ny][nx] = True
                                stack.append((nx, ny))
                    if len(group) >= 4:
                        groups.append(group)
        return groups

def rotation_offset(orientation):
    if orientation == 0:
        return (0, -1)
    elif orientation == 1:
        return (1, 0)
    elif orientation == 2:
        return (0, 1)
    elif orientation == 3:
        return (-1, 0)

class Piece:
    def __init__(self, board):
        self.orientation = 0
        self.pivot = [board.cols // 2, 0]
        offset = rotation_offset(self.orientation)
        self.sub = [self.pivot[0] + offset[0], self.pivot[1] + offset[1]]
        self.colors = (random.choice(PUYO_COLORS), random.choice(PUYO_COLORS))
        self.fall_offset = 0.0

    def move(self, dx, dy, board):
        new_pivot = [self.pivot[0] + dx, self.pivot[1] + dy]
        new_sub = [self.sub[0] + dx, self.sub[1] + dy]
        if board.can_move(new_pivot[0], new_pivot[1]) and board.can_move(new_sub[0], new_sub[1]):
            self.pivot = new_pivot
            self.sub = new_sub
            return True
        return False

    def rotate(self, direction, board):
        new_orientation = (self.orientation + (1 if direction > 0 else -1)) % 4
        offset = rotation_offset(new_orientation)
        new_sub = [self.pivot[0] + offset[0], self.pivot[1] + offset[1]]
        if board.can_move(new_sub[0], new_sub[1]):
            self.orientation = new_orientation
            self.sub = new_sub
        else:
            for kick in [(1, 0), (-1, 0)]:
                new_pivot = [self.pivot[0] + kick[0], self.pivot[1] + kick[1]]
                new_sub = [new_pivot[0] + offset[0], new_pivot[1] + offset[1]]
                if board.can_move(new_pivot[0], new_pivot[1]) and board.can_move(new_sub[0], new_sub[1]):
                    self.pivot = new_pivot
                    self.orientation = new_orientation
                    self.sub = new_sub
                    break

# --- 各種補助関数 ---
def spawn_piece(board):
    piece = Piece(board)
    for pos in [piece.pivot, piece.sub]:
        x, y = pos
        if y >= 0 and not board.can_move(x, y):
            return None
    return piece

def lock_piece(piece, board):
    piece.fall_offset = 0.0
    x, y = piece.pivot
    if y >= 0:
        board.add_puyo(x, y, piece.colors[0])
    x, y = piece.sub
    if y >= 0:
        board.add_puyo(x, y, piece.colors[1])

def draw_board(screen, board):
    for y in range(board.rows):
        for x in range(board.cols):
            rect = pygame.Rect(OFFSET_X + x * CELL_SIZE, OFFSET_Y + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1)
            color = board.grid[y][x]
            if color is not None:
                center = (OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2,
                          OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2)
                radius = CELL_SIZE // 2 - 4
                draw_cute_puyo(screen, center, radius, color)

def draw_piece(screen, piece):
    for puyo, color in zip([piece.pivot, piece.sub], piece.colors):
        x, y = puyo
        draw_x = x * CELL_SIZE + OFFSET_X
        draw_y = y * CELL_SIZE + OFFSET_Y + piece.fall_offset
        center = (draw_x + CELL_SIZE // 2, draw_y + CELL_SIZE // 2)
        radius = CELL_SIZE // 2 - 4
        draw_cute_puyo(screen, center, radius, color)

def animate_removal(screen, groups, start_time, delay):
    current_time = pygame.time.get_ticks()
    elapsed = current_time - start_time
    fraction = min(elapsed / delay, 1)
    for group in groups:
        for (x, y) in group:
            center = (OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2,
                      OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2)
            radius = int((CELL_SIZE // 2 - 4) * (1 - fraction))
            if radius > 0:
                pygame.draw.circle(screen, WHITE, center, radius)

def draw_game_over(screen, font):
    text = font.render("Game Over! Press any key to restart.", True, WHITE)
    rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, rect)

def show_instructions(screen, font):
    screen.fill(BLACK)
    instructions = [
        "Controls:",
        "Left Arrow  : Move Left",
        "Right Arrow : Move Right",
        "Down Arrow  : Fast Drop",
        "Z           : Rotate Left",
        "X           : Rotate Right",
        "",
        "Press any key to start..."
    ]
    offset_y = 100
    for i, line in enumerate(instructions):
        text = font.render(line, True, WHITE)
        rect = text.get_rect(center=(SCREEN_WIDTH // 2, offset_y + i * 30))
        screen.blit(text, rect)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def draw_falling_objects(screen, falling_objects):
    for obj in falling_objects:
        x = obj["x"]
        cy = int(obj["current_y"])
        center = (OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2, cy + CELL_SIZE // 2)
        radius = CELL_SIZE // 2 - 4
        draw_cute_puyo(screen, center, radius, obj["color"])

def draw_next_piece(screen, piece):
    preview_rect = pygame.Rect(SCREEN_WIDTH - 120, 20, 100, 100)
    pygame.draw.rect(screen, GRAY, preview_rect, 2)
    if piece is not None:
        pivot_pos = (preview_rect.centerx, preview_rect.centery)
        offset = rotation_offset(piece.orientation)
        sub_pos = (preview_rect.centerx + offset[0] * CELL_SIZE, preview_rect.centery + offset[1] * CELL_SIZE)
        radius = CELL_SIZE // 2 - 4
        draw_cute_puyo(screen, pivot_pos, radius, piece.colors[0])
        draw_cute_puyo(screen, sub_pos, radius, piece.colors[1])

def draw_score(screen, font, score):
    text = font.render("Score: " + str(score), True, WHITE)
    screen.blit(text, (SCREEN_WIDTH - 120, 140))

def initiate_falling(board):
    falling_objects = []
    for x in range(board.cols):
        non_empty = []
        for y in range(board.rows):
            if board.grid[y][x] is not None:
                non_empty.append((y, board.grid[y][x]))
        new_start = board.rows - len(non_empty)
        for index, (orig_y, color) in enumerate(non_empty):
            final_y = new_start + index
            if final_y > orig_y:
                start_pixel = OFFSET_Y + orig_y * CELL_SIZE
                target_pixel = OFFSET_Y + final_y * CELL_SIZE
                falling_objects.append({
                    "x": x,
                    "current_y": start_pixel,
                    "target_y": target_pixel,
                    "color": color
                })
                board.grid[orig_y][x] = None
    return falling_objects

# --- メインループ ---
def main():
    # ユーザーから盤面サイズを取得（入力がなければデフォルト 12×6）
    rows, cols = get_board_size()
    # 基準サイズの計算（縮尺前）
    base_board_width = cols * BASE_CELL_SIZE
    base_board_height = rows * BASE_CELL_SIZE
    base_screen_width = base_board_width + EXTRA_X
    base_screen_height = base_board_height + EXTRA_Y

    pygame.init()
    # デスクトップ解像度の取得（80%以内に収める）
    info = pygame.display.Info()
    desktop_width = info.current_w
    desktop_height = info.current_h
    scale = min((desktop_width * 0.8 - EXTRA_X) / base_board_width,
                (desktop_height * 0.8 - EXTRA_Y) / base_board_height,
                1.0)
    global CELL_SIZE, BOARD_ROWS, BOARD_COLS, BOARD_WIDTH, BOARD_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT
    CELL_SIZE = int(BASE_CELL_SIZE * scale)
    BOARD_ROWS = rows
    BOARD_COLS = cols
    BOARD_WIDTH = BOARD_COLS * CELL_SIZE
    BOARD_HEIGHT = BOARD_ROWS * CELL_SIZE
    SCREEN_WIDTH = BOARD_WIDTH + EXTRA_X
    SCREEN_HEIGHT = BOARD_HEIGHT + EXTRA_Y

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Puyo Puyo - Cute Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    show_instructions(screen, font)

    board = Board(BOARD_ROWS, BOARD_COLS)
    next_piece = spawn_piece(board)
    if next_piece is None:
        draw_game_over(screen, font)
        pygame.display.flip()
        pygame.time.wait(2000)
        return
    current_piece = next_piece
    next_piece = spawn_piece(board)

    state = "playing"  # "playing", "removing", "falling", "gameover"
    falling_objects = []
    chain_groups = []
    chain_anim_start = 0
    chain_delay = 300
    chain_count = 0
    score = 0

    pygame.key.set_repeat(150, 50)

    while True:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if state == "playing" and current_piece is not None:
                    if event.key == pygame.K_LEFT:
                        current_piece.move(-1, 0, board)
                    elif event.key == pygame.K_RIGHT:
                        current_piece.move(1, 0, board)
                    elif event.key == pygame.K_z:
                        current_piece.rotate(-1, board)
                    elif event.key == pygame.K_x:
                        current_piece.rotate(1, board)
                elif state == "gameover":
                    main()
                    return

        if state == "playing" and current_piece is not None:
            dt_sec = dt / 1000.0
            keys = pygame.key.get_pressed()
            effective_fall_speed = FAST_FALL_SPEED if keys[pygame.K_DOWN] else NORMAL_FALL_SPEED
            potential_offset = current_piece.fall_offset + effective_fall_speed * dt_sec

            allowed_offset = CELL_SIZE
            for cell in [current_piece.pivot, current_piece.sub]:
                cx, cy = cell
                if cy >= 0:
                    if cy == BOARD_ROWS - 1 or board.grid[cy+1][cx] is not None:
                        allowed_offset = 0
                        break

            current_piece.fall_offset = min(potential_offset, allowed_offset)

            if allowed_offset == 0:
                lock_piece(current_piece, board)
                current_piece = None
                chain_groups = board.find_groups()
                if chain_groups:
                    state = "removing"
                    chain_anim_start = pygame.time.get_ticks()
                    chain_count = 1
                else:
                    falling_objects = initiate_falling(board)
                    if falling_objects:
                        state = "falling"
                    else:
                        current_piece = next_piece
                        next_piece = spawn_piece(board)
                        if current_piece is None:
                            state = "gameover"
            else:
                while current_piece.fall_offset >= CELL_SIZE:
                    if current_piece.move(0, 1, board):
                        current_piece.fall_offset -= CELL_SIZE
                    else:
                        current_piece.fall_offset = 0.0
                        lock_piece(current_piece, board)
                        current_piece = None
                        chain_groups = board.find_groups()
                        if chain_groups:
                            state = "removing"
                            chain_anim_start = pygame.time.get_ticks()
                            chain_count = 1
                        else:
                            falling_objects = initiate_falling(board)
                            if falling_objects:
                                state = "falling"
                            else:
                                current_piece = next_piece
                                next_piece = spawn_piece(board)
                                if current_piece is None:
                                    state = "gameover"
                        break

        elif state == "removing":
            current_time = pygame.time.get_ticks()
            if current_time - chain_anim_start >= chain_delay:
                points = 0
                for group in chain_groups:
                    points += (len(group) ** 2) * chain_count
                score += points
                for group in chain_groups:
                    for (x, y) in group:
                        board.remove_puyo(x, y)
                falling_objects = initiate_falling(board)
                if falling_objects:
                    state = "falling"
                else:
                    new_groups = board.find_groups()
                    if new_groups:
                        chain_groups = new_groups
                        chain_anim_start = current_time
                        chain_count += 1
                        state = "removing"
                    else:
                        current_piece = next_piece
                        next_piece = spawn_piece(board)
                        if current_piece is None:
                            state = "gameover"
                        else:
                            state = "playing"
                            chain_count = 0

        elif state == "falling":
            for obj in falling_objects:
                obj["current_y"] += NORMAL_FALL_SPEED * (dt / 1000.0)
                if obj["current_y"] >= obj["target_y"]:
                    obj["current_y"] = obj["target_y"]
            if all(obj["current_y"] >= obj["target_y"] for obj in falling_objects):
                for obj in falling_objects:
                    final_row = int((obj["target_y"] - OFFSET_Y) / CELL_SIZE)
                    board.grid[final_row][obj["x"]] = obj["color"]
                falling_objects = []
                new_groups = board.find_groups()
                if new_groups:
                    chain_groups = new_groups
                    chain_anim_start = pygame.time.get_ticks()
                    chain_count += 1
                    state = "removing"
                else:
                    current_piece = next_piece
                    next_piece = spawn_piece(board)
                    if current_piece is None:
                        state = "gameover"
                    else:
                        state = "playing"

        screen.fill(BLACK)
        draw_board(screen, board)
        if state == "playing" and current_piece is not None:
            draw_piece(screen, current_piece)
        if state == "removing":
            animate_removal(screen, chain_groups, chain_anim_start, chain_delay)
        if state == "falling":
            draw_falling_objects(screen, falling_objects)
        if state == "gameover":
            draw_game_over(screen, font)
        draw_next_piece(screen, next_piece)
        draw_score(screen, font, score)
        pygame.display.flip()

if __name__ == "__main__":
    main()
