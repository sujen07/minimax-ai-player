import time
import chess
import minimax
import pygame
pygame.init()

size = (600, 600)  # Size of the window
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Chess Game")

def load_pieces():
    pieces = {}
    for name in ['Pw', 'Nw', 'Bw', 'Rw', 'Qw', 'Kw', 'Pb', 'Nb', 'Bb', 'Rb', 'Qb', 'Kb']:
        if name[-1] == 'w':
            piece_name = name[0].upper()
        else:
            piece_name = name[0].lower()
        image = pygame.image.load(f'pieces-basic-svg/{name}.svg')
        pieces[piece_name] = pygame.transform.scale(image, (75, 75))
        
    return pieces

pieces = load_pieces()

def draw_dotted_square(screen, col, row, color):
    square = pygame.Rect(col*75, row*75, 75, 75)
    for i in range(0, 75, 4):  # Adjust the range and step for different dot sizes and spacing
        pygame.draw.line(screen, color, (square.left + i, square.top), (square.left + i, square.bottom), 1)
        pygame.draw.line(screen, color, (square.left, square.top + i), (square.right, square.top + i), 1)

def draw_board_and_pieces(screen, board, pieces, selected_piece, blink_timer, last_ai_move, possible_moves):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    highlight_color = pygame.Color("blue")  # Color for highlighting the selected piece
    check_color = pygame.Color("red")       # Color for highlighting the king in check
    blink_colors = [pygame.Color("red"), pygame.Color("blue")]  # Colors for blinking
    move_highlight_color = pygame.Color("green")

    # Draw squares on the board
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            square = pygame.Rect(c*75, r*75, 75, 75)
            pygame.draw.rect(screen, color, square)

            # Highlight the selected piece
            if selected_piece is not None and selected_piece == chess.square(c, r):
                pygame.draw.rect(screen, highlight_color, square, 5)

                # Blink effect for the king in check
                if board.is_check() and board.piece_at(selected_piece).piece_type == chess.KING:
                    blink_color = blink_colors[blink_timer % 2]  # Alternate colors
                    pygame.draw.rect(screen, blink_color, square, 5)

            # Highlight the king if in check
            elif board.is_check():
                king_pos = board.king(board.turn)
                if king_pos is not None and king_pos == chess.square(c, r):
                    pygame.draw.rect(screen, check_color, square, 5)

    # Draw pieces on the board
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            screen.blit(pieces[piece.symbol()], (col*75, row*75))
    if last_ai_move:
        start_square = last_ai_move.from_square
        end_square = last_ai_move.to_square
        for square in [start_square, end_square]:
            row, col = divmod(square, 8)
            pygame.draw.rect(screen, pygame.Color("yellow"), pygame.Rect(col*75, row*75, 75, 75), 5)
    if possible_moves:
        for move in possible_moves:
            end_square = move.to_square
            row, col = divmod(end_square, 8)
            draw_dotted_square(screen, col, row, move_highlight_color)



def run_game(player=chess.WHITE):
    last_ai_move = None
    board = chess.Board()
    # Initialize chess board
    board = chess.Board()

    # Variables to keep track of the game state
    selected_piece = None  # Store the square of the selected piece
    possible_moves = []    # Store possible moves for the selected piece
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Check if its AI turn
            if player != board.turn:
                time.sleep(0.5)
                start = time.time()
                move = minimax.minimax(board)
                board.push(move)
                last_ai_move = move
                end_time = time.time()
                print('AI move time: ', end_time - start)
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position and convert it to board coordinates
                x, y = event.pos
                col = x // 75
                row = y // 75
                square = chess.square(col, row)

                if selected_piece is None:
                    piece = board.piece_at(square)
                    if piece and piece.color == board.turn:
                        selected_piece = square
                        possible_moves = [move for move in board.legal_moves if move.from_square == selected_piece]
                else:
                    # Move the piece
                    if board.piece_at(selected_piece).piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                        # Automatically promote to a queen for simplicity
                        move = chess.Move(selected_piece, square, promotion=chess.QUEEN)
                    else:
                        move = chess.Move(selected_piece, square)
                    if move in possible_moves:
                        board.push(move)
                    selected_piece = None
                    possible_moves = []
            if board.is_checkmate():
                print("Checkmate! Game over.")
                running = False  # or handle the end of the game as you prefer
            elif board.is_stalemate():
                print("Stalemate! Game over.")
                running = False
        # Draw the board and pieces
        draw_board_and_pieces(screen, board, pieces, selected_piece, blink_timer=1, last_ai_move=last_ai_move, possible_moves=possible_moves)

        # Update the display
        pygame.display.flip()


def restart_and_select_screen(screen):
    font = pygame.font.Font(None, 36)
    running = True

    # Define button rectangles
    play_button = pygame.Rect(200, 150, 200, 50)
    exit_button = pygame.Rect(200, 250, 200, 50)
    white_button = pygame.Rect(150, 350, 100, 50)
    black_button = pygame.Rect(350, 350, 100, 50)

    while running:
        # Clear the screen with a background color (e.g., white)
        screen.fill(pygame.Color("white"))

        # Draw buttons
        pygame.draw.rect(screen, pygame.Color("grey"), play_button)
        pygame.draw.rect(screen, pygame.Color("grey"), exit_button)
        pygame.draw.rect(screen, pygame.Color("grey"), white_button)
        pygame.draw.rect(screen, pygame.Color("grey"), black_button)

        # Add text to buttons
        play_text = font.render('Play', True, pygame.Color('Black'))
        exit_text = font.render('Exit', True, pygame.Color('Black'))
        white_text = font.render('White', True, pygame.Color('Black'))
        black_text = font.render('Black', True, pygame.Color('Black'))

        screen.blit(play_text, (play_button.x + 70, play_button.y + 10))
        screen.blit(exit_text, (exit_button.x + 70, exit_button.y + 10))
        screen.blit(white_text, (white_button.x + 20, white_button.y + 10))
        screen.blit(black_text, (black_button.x + 20, black_button.y + 10))

        # Update the display
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if play_button.collidepoint(x, y):
                    return True, None
                elif exit_button.collidepoint(x, y):
                    pygame.quit()
                    exit()
                elif white_button.collidepoint(x, y):
                    return True, chess.WHITE
                elif black_button.collidepoint(x, y):
                    return True, chess.BLACK

    return False, None


def main():
    while True:
        restart, player = restart_and_select_screen(screen)
        if not restart:
            break
        if not player:
            run_game()
        else:
            run_game(player=player)

if __name__ == '__main__':
    main()