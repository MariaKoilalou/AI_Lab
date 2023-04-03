import time


# Choose whether you play with 'X' or 'Ο'
def choose_player():
    player = input("Do you want to be X or O? ").upper()
    while player not in ['X', 'O']:
        print("Invalid input. Please choose X or O.")
        player = input("Do you want to be X or O? ").upper()
    if player == 'X':
        return 'X', 'O'
    else:
        return 'O', 'X'


# Constants for player and computer moves
EMPTY = ' '
PLAYER, COMPUTER = choose_player()


# Function to draw the board
def draw_board(board):
    print('╔═══════════════╗')
    print('║  ' + board[0][0] + '  | ' + board[0][1] + ' | ' + board[0][2] + '   ║')
    print('║  -----------  ║')
    print('║  ' + board[1][0] + '  | ' + board[1][1] + ' | ' + board[1][2] + '   ║')
    print('║  -----------  ║')
    print('║  ' + board[2][0] + '  | ' + board[2][1] + ' | ' + board[2][2] + '   ║')
    print('╚═══════════════╝')


# Function to check if a move is valid
def is_valid_move(board, row, col):
    # Check if row and col are integers
    if not isinstance(row, int) or not isinstance(col, int):
        return False

    # Check if the row and col are within the bounds of the board
    if row < 0 or row >= len(board) or col < 0 or col >= len(board[0]):
        return False

    # Check if the specified cell is empty
    if board[row][col] == EMPTY:
        return True

    # Otherwise, the move is not valid
    return False



# Function to make a move on the board
def make_move(board, row, col, player):
    board[row][col] = player


# Function to check if the game is over
def is_game_over(board):
    # Check rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != EMPTY:
            return True
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != EMPTY:
            return True
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return True
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return True
    # Check for a tie
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                return False
    return True


# Function to get the score for a board
def get_score(board):
    # Check rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2]:
            if board[row][0] == COMPUTER:
                return 1
            elif board[row][0] == PLAYER:
                return -1
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            if board[0][col] == COMPUTER:
                return 1
            elif board[0][col] == PLAYER:
                return -1
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == COMPUTER:
            return 1
        elif board[0][0] == PLAYER:
            return -1
    if board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == COMPUTER:
            return 1
        elif board[0][2] == PLAYER:
            return -1
    # No winner yet
    return 0


# Function to get the best move for the computer using
def get_best_move(board, depth, alpha, beta, player):
    # Check if the game is over or the maximum depth has been reached
    if is_game_over(board) or depth == 0:
        return None, get_score(board)
    # Set the best move and score
    best_move = None
    if player == COMPUTER:
        best_score = -float('inf')
    else:
        best_score = float('inf')
    # Loop through all possible moves
    for row in range(3):
        for col in range(3):
            if is_valid_move(board, row, col):
                # Make the move
                make_move(board, row, col, player)

                # Recursively get the score for the move
                _, score = get_best_move(board, depth - 1, alpha, beta, PLAYER if player == COMPUTER else COMPUTER)

                # Undo the move
                make_move(board, row, col, ' ')

                # Update the best score and move
                if player == COMPUTER:
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
                else:
                    if score < best_score:
                        best_score = score
                        best_move = (row, col)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break

    # Return the best move and score
    return best_move, best_score


def play_game():
    # Initialize the board
    board = [[' ' for _ in range(3)] for _ in range(3)]
    if PLAYER == 'O':
        # Computer plays first
        print('Computer is thinking...')
        time.sleep(1)
        move, _ = get_best_move(board, 4, -float('inf'), float('inf'), COMPUTER)
        make_move(board, move[0], move[1], COMPUTER)
    # Loop until the game is over
    while not is_game_over(board):
        draw_board(board)
        # Player's turn
        row = input('Enter row (1-3): ')
        col = input('Enter column (1-3): ')
        if row.isdigit() and col.isdigit():
            row = int(row)-1
            col = int(col)-1
        else:
            print("Please enter an integer.")
        if is_valid_move(board, row, col):
            make_move(board, row, col, PLAYER)
            draw_board(board)
        else:
            print('Invalid move!')
            continue
        if is_game_over(board):
            break
        # Computer's turn
        print('Computer is thinking...')
        time.sleep(1)
        move, _ = get_best_move(board, 4, -float('inf'), float('inf'), COMPUTER)
        make_move(board, move[0], move[1], COMPUTER)

    # Game is over
    draw_board(board)
    score = get_score(board)
    if score > 0:
        print('Computer wins!')
    elif score < 0:
        print('Player wins!')
    else:
        print('Tie game!')



if __name__ == '__main__':
    play_game()
