"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def heuristic_open_move(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))

def heuristic_improved(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

def heuristic_center(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y)**2 + (w - x)**2)

def heuristic1(game, player, weight = 2, center_weight = 2):
    # Custom Heuristic
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # We have moves to play. How many more than our opponent?
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    player_moves_left = len(player_moves)
    opponent_moves_left = len(opponent_moves)

    center_y_pos, center_x_pos = int(game.height / 2), int(game.width / 2)
    player_y_pos, player_x_pos = game.get_player_location(player)
    opponent_y_pos, opponent_x_pos = game.get_player_location(game.get_opponent(player))
    player_distance = abs(player_y_pos - center_y_pos) + abs(player_x_pos - center_x_pos)
    opponent_distance = abs(opponent_y_pos - center_y_pos) + abs(opponent_x_pos - center_x_pos)

    if (center_y_pos, center_x_pos) in player_moves:
        w, h = game.width / 2., game.height / 2.
        y, x = game.get_player_location(player)
        return float((h - y) ** 2 + (w - x) ** 2)
    else:
        initial_moves_available = float(game.width * game.height)
        num_blank_spaces = len(game.get_blank_spaces())
        decay_factor = num_blank_spaces / initial_moves_available
        opponent_weight, player_weight = weight, 1

        for move in player_moves:
            if move[0] == center_y_pos or move[1] == center_x_pos:
                # if player_moves_left <= 2:
                #     player_weight *= (center_weight * decay_factor)
                # else:
                player_weight *= (center_weight * decay_factor)

        for move in opponent_moves:
            if move[0] == center_y_pos or move[1] == center_x_pos:
                # if opponent_moves_left >= 6:
                #     opponent_weight *= (center_weight * decay_factor)
                # else:
                opponent_weight *= (center_weight * decay_factor)

        return float((player_moves_left * player_weight) - (opponent_moves_left * opponent_weight))

def heuristic2(game, player, weight = 2, center_weight = 2):
    # Weighted Center
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_y_pos, center_x_pos = int(game.height / 2), int(game.width / 2)

    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    player_moves_left = len(player_moves)
    opponent_moves_left = len(opponent_moves)

    opponent_weight, player_weight = weight, 1

    for move in player_moves:
        if move[0] == center_y_pos or move[1] == center_x_pos:
            player_weight *= center_weight

    for move in opponent_moves:
        if move[0] == center_y_pos or move[1] == center_x_pos:
            opponent_weight *= center_weight

    return float((player_moves_left * player_weight) - (opponent_moves_left * opponent_weight))

def heuristic3(game, player, weight = 2, center_weight = 2):
    # Decay weighted center
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_y_pos, center_x_pos = int(game.height / 2), int(game.width / 2)

    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    player_moves_left = len(player_moves)
    opponent_moves_left = len(opponent_moves)

    opponent_weight, player_weight = weight, 1

    initial_moves_available = float(game.width * game.height)
    num_blank_spaces = len(game.get_blank_spaces())
    decay_factor = num_blank_spaces / initial_moves_available

    for move in player_moves:
        if move[0] == center_y_pos or move[1] == center_x_pos:
            player_weight *= (center_weight * decay_factor)

    for move in opponent_moves:
        if move[0] == center_y_pos or move[1] == center_x_pos:
            opponent_weight *= (center_weight * decay_factor)

    return float((player_moves_left * player_weight) - (opponent_moves_left * opponent_weight))

def custom_score(game, player, weight = 2, center_weight = 2):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return heuristic2(game, player)


def custom_score_2(game, player, weight = 2, center_weight = 2):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return heuristic3(game, player)


def custom_score_3(game, player, weight = 2, center_weight = 2):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return heuristic1(game, player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        # def _recursive_minimax(self, game, depth, isMaximizing=True):
        #     # Timeout has occurred and an exception is raised.
        #     if self.time_left() < self.TIMER_THRESHOLD:
        #         raise SearchTimeout()
        #
        #     # If the depth = 0 then we have reached the top of the game tree and should return the score.
        #     if depth == 0:
        #         return self.score(game, self)
        #
        #     # Retrieve all the legal moves associated with the current position.
        #     legal_moves = game.get_legal_moves()
        #
        #     # If there are no legal moves then return score to indicate no legal moves remain.
        #     if not legal_moves:
        #         if isMaximizing:
        #             return float("-inf")
        #         else:
        #             return float("inf")
        #
        #     # Maximizing case: Recursively traverse down the game tree and determine best move before the time runs out.
        #     if isMaximizing:
        #         best_score = float("-inf")
        #         for move in legal_moves:
        #             score = _recursive_minimax(self, game.forecast_move(move), depth - 1, False)
        #             best_score = max(best_score, score)
        #         return best_score
        #     else:
        #         # Minimizing case: Recursively traverse down the game tree and determine the bes move that will reduce
        #         # the opponent's chances of winning.
        #         best_score = float("inf")
        #         for move in legal_moves:
        #             score = _recursive_minimax(self, game.forecast_move(move), depth - 1, True)
        #             best_score = min(best_score, score)
        #         return best_score
        #
        # # Retrieve all the legal moves associated with the current position.
        # legal_moves = game.get_legal_moves()
        #
        # # If there are no legal moves then return -1, -1 to indicate no legal moves remain.
        # if not legal_moves:
        #     return (-1, -1)
        #
        # # Store the moves.
        # moves_map = {}
        #
        # # for all the moves, recursively call minimax function to retrieve all the best moves before time runs out.
        # for move in legal_moves:
        #     moves_map[move] = _recursive_minimax(self, game.forecast_move(move), depth - 1, False)
        #
        # # Determine the best move from the list of moves.
        # best_move = max(moves_map, key=lambda k: moves_map[k])
        #
        # return best_move

        def max_value(self, game, depth):
            """ Return the value for a loss (-1) if the game is over, otherwise return the maximum value over all
            legal child nodes.
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout

            if not bool(game.get_legal_moves()):
                return -1  # by assumption #2

            # Stop and return score after running out of depth
            if (depth == 0):
                return self.score(game, self)

            v = float("-inf")

            for m in game.get_legal_moves():
                v = max(v, min_value(self, game.forecast_move(m), depth - 1))

            return v

        def min_value(self, game, depth):
            """ Return the value for a loss (1) if the game is over, otherwise return the minimum value over all
            legal child nodes.
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout

            if not bool(game.get_legal_moves()):
                return 1  # by assumption #2

            # Stop and return score after running out of depth
            if (depth == 0):
                return self.score(game, self)

            v = float("inf")

            for m in game.get_legal_moves():
                v = min(v, max_value(self, game.forecast_move(m), depth - 1))

            return v

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()

        if not moves:
            return (-1, -1)

        best_score = float("-inf")
        best_move = moves[0]

        for move in moves:
            score = min_value(self, game.forecast_move(move), depth - 1)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            # Use Iterative Deepening with Alpha Beta Pruning to find best move with least number of tries.
            depth = 0

            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # def _recursive_alphabeta(self, game, depth, alpha, beta, isMaximizing=True):
        #     # Timeout has occurred and an exception is raised.
        #     if self.time_left() < self.TIMER_THRESHOLD:
        #         raise SearchTimeout()
        #
        #     # If the depth = 0 then we have reached the top of the game tree and should return the score.
        #     if depth == 0:
        #         return self.score(game, self)
        #
        #     # Retrieve all the legal moves associated with the current position.
        #     legal_moves = game.get_legal_moves()
        #
        #     # If there are no legal moves then return score to indicate no legal moves remain.
        #     if not legal_moves:
        #         if isMaximizing:
        #             return -1
        #         else:
        #             return 1
        #
        #     # Maximizing case: Recursively traverse down the game tree and determine best move before the time runs out.
        #     if isMaximizing:
        #         best_score = float("-inf")
        #         score = float("-inf")
        #         for move in legal_moves:
        #             score = _recursive_alphabeta(self, game.forecast_move(move), depth - 1, alpha, beta, False)
        #             best_score = max(best_score, score)
        #             if score >= beta:
        #                 return score
        #             alpha = max(alpha, best_score)
        #
        #         return best_score
        #     else:
        #         # Minimizing case: Recursively traverse down the game tree and determine the bes move that will reduce
        #         # the opponent's chances of winning.
        #         best_score = float("inf")
        #         score = float("inf")
        #         for move in legal_moves:
        #             score = _recursive_alphabeta(self, game.forecast_move(move), depth - 1, alpha, beta, True)
        #             best_score = min(best_score, score)
        #             if score <= alpha:
        #                 return score
        #             beta = min(beta, best_score)
        #
        #         return best_score
        #
        # # Retrieve all the legal moves associated with the current position.
        # legal_moves = game.get_legal_moves()
        #
        # # If there are no legal moves then return -1, -1 to indicate no legal moves remain.
        # if not legal_moves:
        #     return (-1, -1)
        #
        # # Store the moves.
        # moves_map = {}
        #
        # # for all the moves, recursively call alphabeta function to retrieve all the best moves before time runs out.
        # for move in legal_moves:
        #     moves_map[move] = _recursive_alphabeta(self, game.forecast_move(move), depth - 1, alpha, beta, False)
        #
        # # Determine the best move from the list of moves.
        # best_move = max(moves_map, key=lambda k: moves_map[k])
        #
        # return best_move

        def max_value(self, game, depth, alpha, beta):
            """ Return the value for a loss (-1) if the game is over, otherwise return the maximum value over all
            legal child nodes.
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout

            imoves = game.get_legal_moves()

            if not bool(moves):
                return -1  # by assumption #2

            # Stop and return score after running out of depth
            if (depth == 0):
                return self.score(game, self)

            v = float("-inf")

            for m in moves:
                v = max(v, min_value(self, game.forecast_move(m), depth - 1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)

            return v

        def min_value(self, game, depth, alpha, beta):
            """ Return the value for a loss (1) if the game is over, otherwise return the minimum value over all
            legal child nodes.
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout

            moves = game.get_legal_moves()

            if not bool(moves):
                return 1  # by assumption #2

            # Stop and return score after running out of depth
            if (depth == 0):
                return self.score(game, self)

            v = float("inf")

            for m in moves:
                v = min(v, max_value(self, game.forecast_move(m), depth - 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)

            return v

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout

        moves = game.get_legal_moves()

        if not moves:
            return (-1, -1)

        best_score = float("-inf")
        best_move = moves[0]

        for move in moves:
            score = min_value(self, game.forecast_move(move), depth - 1, alpha, beta)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        return best_move
