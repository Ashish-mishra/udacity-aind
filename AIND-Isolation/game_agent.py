"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """ Custom Heuristic #1
    The difference in the number of available moves between the current
    player and its opponent one ply ahead in the future is used as the
    score of the current game state.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    opp = game.get_opponent(player)
    opp_moves = game.get_legal_moves(opp)
    p_moves = game.get_legal_moves()
    own_score = 0
    opp_score = 0
    for move in p_moves:
        own_score += len(game.forecast_move(move).get_legal_moves())

    for move in opp_moves:
        opp_score += len(game.forecast_move(move).get_legal_moves())

    return float(own_score - opp_score + len(p_moves) - len(opp_moves))


def custom_score_2(game, player):
    """ Custom Heuristic #2
    This is a heuristic similar to the base heuristic (IM Improved) but more aggressive in the sense that
    it weights the opponents legal moves in a ratio to player's legals moves before taking the difference.
    """
    opp = game.get_opponent(player)
    opp_moves = game.get_legal_moves(opp)
    p_moves = game.get_legal_moves()
    if not opp_moves:
        return float("inf")
    if not p_moves:
        return float("-inf")

    return float(len(p_moves) - 2*(len(opp_moves)))
    

def custom_score_3(game, player):
    """ Custom Heuristic #3
    This heuristic calculates the ratio of player moves to opponent moves.
    """
    opp = game.get_opponent(player)
    opp_moves = game.get_legal_moves(opp)
    p_moves = game.get_legal_moves()
    if not opp_moves:
        return float("inf")
    if not p_moves:
        return float("-inf")

    return float(len(p_moves)/len(opp_moves))


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
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        return self.minimax_move(game, depth)[0]

    def active_player(self, game):
        return game.active_player == self
    
    def minimax_move(self, game, depth):
        """
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth == 0:
            return (game.get_player_location(self), self.score(game, self))

        value, func, best_move = None, None, (-1, -1)
        
        if self.active_player(game):
            func, value = max, float("-inf")
        else:
            func, value = min, float("inf")

        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move)
            score = self.minimax_move(next_ply, depth - 1)[1]
            if func(value, score) == score:
                best_move = move
                value = score

        return (best_move, value)

    


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

        # TODO: finish this function!
        # raise NotImplementedError
        move = (-1, -1)
        for i in range(1, 10000):
            try:
                move = self.alphabeta(game, i)
            except SearchTimeout:
                break
        return move


    def active_player(self, game):
        return game.active_player == self

    

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

        # TODO: finish this function!
        best_score = float("-inf")
        best_move = (-1,-1)
        for m in game.get_legal_moves():
            v=self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            if v > best_score:
                best_score = v
                best_move = m
            alpha = max(alpha, v)
        return best_move

    def terminal_test(self, gameState, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return not bool(gameState.get_legal_moves()) or depth<=0

    

    def min_value(self, gameState, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(gameState, depth):
            return self.score(gameState, self)

        v=float("inf")

        for m in gameState.get_legal_moves():
            v = min(v, self.max_value(gameState.forecast_move(m), depth - 1, alpha, beta))
            if v<= alpha:
                return v
            beta = min(beta, v)

        return v


    def max_value(self, gameState, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(gameState, depth):
            return self.score(gameState, self)

        v=float("-inf")

        for m in gameState.get_legal_moves():
            v = max(v, self.min_value(gameState.forecast_move(m), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v
    
