import pygame
import random

# 初期化
pygame.init()

# 画面サイズ
s_width = 300
s_height = 600
block_size = 30

play_width = block_size * 10
play_height = block_size * 20

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

# 色定義
colors = [
    (0, 0, 0),
    (255, 0, 0),     # 赤
    (0, 255, 0),     # 緑
    (0, 0, 255),     # 青
    (255, 255, 0),   # 黄
    (255, 165, 0),   # オレンジ
    (128, 0, 128),   # 紫
    (0, 255, 255)    # シアン
]

# テトロミノの形
S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [colors[1], colors[2], colors[3], colors[4], colors[5], colors[6], colors[7]]

class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0

def create_grid(locked_positions={}):
    grid = [[colors[0] for _ in range(10)] for _ in range(20)]
    for (x, y), color in locked_positions.items():
        grid[y][x] = color
    return grid

def convert_shape_format(piece):
    positions = []
    format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((piece.x + j - 2, piece.y + i - 4))
    return positions

def valid_space(piece, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == colors[0]] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(piece)
    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False
    return True

def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False

def get_shape():
    return Piece(5, 0, random.choice(shapes))

def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("comicsans", size, bold=True)
    label = font.render(text, 1, color)
    surface.blit(label, (top_left_x + play_width /2 - label.get_width()/2, top_left_y + play_height /2 - label.get_height()/2))

def draw_grid(surface, grid):
    for i in range(len(grid)):
        pygame.draw.line(surface, (128,128,128), (top_left_x, top_left_y + i*block_size), (top_left_x + play_width, top_left_y + i * block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128,128,128), (top_left_x + j*block_size, top_left_y), (top_left_x + j*block_size, top_left_y + play_height))

def clear_rows(grid, locked):
    increment = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if colors[0] not in row:
            increment +=1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j,i)]
                except:
                    continue
    if increment >0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + increment)
                locked[newKey] = locked.pop(key)
    return increment

def draw_next_shape(piece, surface):
    font = pygame.font.SysFont("comicsans", 30)
    label = font.render('Next Shape', 1, (255,255,255))
    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100
    format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, piece.color, (sx + j*block_size, sy + i*block_size, block_size, block_size), 0)

    surface.blit(label, (sx + 10, sy - 30))

def update_score(nscore):
    score = nscore
    return score

def main():
    global top_left_x, top_left_y, play_width, play_height, block_size
    locked_positions = {}
    grid = create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.5
    level_time = 0
    score = 0

    while run:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        if fall_time/1000 >= fall_speed:
            fall_time = 0
            current_piece.y +=1
            if not (valid_space(current_piece, grid)) and current_piece.y >0:
                current_piece.y -=1
                change_piece = True

        if level_time/1000 > 5:
            level_time =0
            if fall_speed > 0.1:
                fall_speed -= 0.05

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -=1
                    if not valid_space(current_piece, grid):
                        current_piece.x +=1
                elif event.key == pygame.K_RIGHT:
                    current_piece.x +=1
                    if not valid_space(current_piece, grid):
                        current_piece.x -=1
                elif event.key == pygame.K_DOWN:
                    current_piece.y +=1
                    if not valid_space(current_piece, grid):
                        current_piece.y -=1
                elif event.key == pygame.K_UP:
                    current_piece.rotation = (current_piece.rotation +1) % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = (current_piece.rotation -1) % len(current_piece.shape)

        shape_pos = convert_shape_format(current_piece)

        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        if change_piece:
            for pos in shape_pos:
                locked_positions[pos] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            cleared = clear_rows(grid, locked_positions)
            if cleared >0:
                score += cleared *10

        draw_window(grid, score)
        draw_next_shape(next_piece, screen)
        pygame.display.update()

        if check_lost(locked_positions):
            run = False
            draw_text_middle(screen, "You Lost!", 40, (255,255,255))
            pygame.display.update()
            pygame.time.delay(2000)

def draw_window(grid, score):
    screen.fill((0,0,0))

    # タイトル
    font = pygame.font.SysFont("comicsans", 60)
    label = font.render("Tetris", 1, (255,255,255))
    screen.blit(label, (top_left_x + play_width /2 - label.get_width()/2, 30))

    # スコア
    font = pygame.font.SysFont("comicsans", 30)
    label = font.render(f"Score: {score}", 1, (255,255,255))
    screen.blit(label, (top_left_x - 200, top_left_y + play_height + 20))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(screen, grid[i][j],
                             (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0)

    draw_grid(screen, grid)
    pygame.draw.rect(screen, (255,0,0), (top_left_x, top_left_y, play_width, play_height), 5)

def main_menu():
    global screen
    screen = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption('Tetris')
    run = True
    while run:
        screen.fill((0,0,0))
        font = pygame.font.SysFont("comicsans", 30)
        label = font.render("Press Any Key to Play", 1, (255,255,255))
        controls = [
            "Controls:",
            "Left Arrow: Move Left",
            "Right Arrow: Move Right",
            "Down Arrow: Move Down",
            "Up Arrow: Rotate"
        ]
        screen.blit(label, (s_width /2 - label.get_width()/2, s_height /2 - label.get_height()/2 - 100))
        for idx, text in enumerate(controls):
            ctrl_label = font.render(text,1, (255,255,255))
            screen.blit(ctrl_label, (s_width /2 - ctrl_label.get_width()/2, s_height /2 - ctrl_label.get_height()/2 -50 + idx *30))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main()
    pygame.quit()

main_menu()
