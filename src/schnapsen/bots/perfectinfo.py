from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Any

from schnapsen.game import (
    Bot,
    Move,
    PlayerPerspective,
    GamePhase,
    GameState,
    FollowerPerspective,
    LeaderPerspective,
    GamePlayEngine,
    SchnapsenTrickScorer,
)


# =============================================================================
# Phase-2: REGULAR AlphaBeta (no cheating) — identical strategy to alphabeta.py
# =============================================================================

class AlphaBetaBot(Bot):
    """
    A bot playing the alphabeta strategy in the second phase of the game.
    It cannot be used for the first phase. Delegate to this in phase 2.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        assert (
            perspective.get_phase() == GamePhase.TWO
        ), "AlphaBetaBot can only work in the second phase of the game."
        _, move = self.value(
            perspective.get_state_in_phase_two(),
            perspective.get_engine(),
            leader_move=leader_move,
            maximizing=True,
        )
        return move

    def value(
        self,
        state: GameState,
        engine: GamePlayEngine,
        leader_move: Optional[Move],
        maximizing: bool,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> tuple[float, Move]:
        my_perspective: PlayerPerspective
        if leader_move is None:
            my_perspective = LeaderPerspective(state, engine)
        else:
            my_perspective = FollowerPerspective(state, engine, leader_move)

        valid_moves = my_perspective.valid_moves()

        best_value = float("-inf") if maximizing else float("inf")
        best_move: Optional[Move] = None

        for move in valid_moves:
            if leader_move is None:
                # we are leader
                value, _ = self.value(
                    state=state,
                    engine=engine,
                    leader_move=move,
                    maximizing=not maximizing,
                    alpha=alpha,
                    beta=beta,
                )
            else:
                # we are follower -> complete trick
                leader = OneFixedMoveBot(leader_move)
                follower = OneFixedMoveBot(move)
                new_game_state = engine.play_one_trick(
                    game_state=state,
                    new_leader=leader,
                    new_follower=follower,
                )

                winning_info = SchnapsenTrickScorer().declare_winner(new_game_state)
                if winning_info:
                    winner = winning_info[0].implementation
                    points = float(winning_info[1])
                    follower_wins = winner == follower

                    if not follower_wins:
                        points = -points
                    if not maximizing:
                        points = -points
                    value = points
                else:
                    leader_stayed = leader == new_game_state.leader.implementation
                    if leader_stayed:
                        next_maximizing = not maximizing
                    else:
                        next_maximizing = maximizing
                    value, _ = self.value(new_game_state, engine, None, next_maximizing, alpha, beta)

            # alphabeta pruning
            if maximizing:
                if value > best_value:
                    best_move = move
                    best_value = value
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            else:
                if value < best_value:
                    best_move = move
                    best_value = value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        assert best_move is not None
        return best_value, best_move


class OneFixedMoveBot(Bot):
    def __init__(self, move: Move) -> None:
        super().__init__(name=None)
        self.first_move: Optional[Move] = move

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        assert self.first_move is not None, "This bot can only play one move, after that it ends"
        move = self.first_move
        self.first_move = None
        return move


# =============================================================================
# Phase-1: PERFECT-INFO AlphaBeta (cheats by peeking full GameState)
# =============================================================================
def _rig_to_all_trumps(perspective: PlayerPerspective) -> None:
    state: GameState = perspective._PlayerPerspective__game_state  # type: ignore[attr-defined]
    trump_suit = state.talon.trump_suit()

    # Identify which BotState is "me" vs "opponent"
    if perspective.am_i_leader():
        me = state.leader
        opp = state.follower
    else:
        me = state.follower
        opp = state.leader

    my_cards: List[Any] = list(me.hand.cards)
    opp_cards: List[Any] = list(opp.hand.cards)
    talon_cards: List[Any] = list(getattr(state.talon, "_cards", []))

    # Pick 5 trump cards (these are unique card objects)
    all_trumps = [c for c in (my_cards + opp_cards + talon_cards) if c.suit == trump_suit]
    if len(all_trumps) < 5:
        return
    in_my_hand = [c for c in my_cards if c.suit == trump_suit]
    not_in_my_hand = [c for c in all_trumps if c not in my_cards]
    target_trumps = (in_my_hand + not_in_my_hand)[:5]

    # Cards we need to steal (target trumps not already in my hand)
    missing_trumps = [t for t in target_trumps if t not in my_cards]

    # Cards we will give back = exactly the cards currently in my hand that are not in target_trumps
    give_back = [c for c in my_cards if c not in target_trumps]

    # In a 5-card hand, these counts must match for a clean 1-for-1 swap
    if len(give_back) != len(missing_trumps):
        # Safety fallback: do nothing rather than break invariants
        return

    # For each missing trump, swap it out of opp or talon with one give-back card
    for t, g in zip(missing_trumps, give_back):
        if t in opp_cards:
            idx = opp_cards.index(t)
            opp_cards[idx] = g
        elif t in talon_cards:
            idx = talon_cards.index(t)
            talon_cards[idx] = g
        else:
            # Shouldn't happen, but don't corrupt state if it does
            return

    # Now my hand is exactly the 5 target trumps (no cards lost, none duplicated)
    me.hand.cards[:] = target_trumps
    opp.hand.cards[:] = opp_cards
    if hasattr(state.talon, "_cards"):
        state.talon._cards[:] = talon_cards  # type: ignore[attr-defined]



def _peek_full_state(perspective: PlayerPerspective) -> GameState:
    # Name-mangled private attribute: PlayerPerspective.__game_state
    return perspective._PlayerPerspective__game_state  # type: ignore[attr-defined]


def _get_engine(perspective: PlayerPerspective) -> GamePlayEngine:
    # Most versions expose get_engine()
    if hasattr(perspective, "get_engine"):
        return perspective.get_engine()  # type: ignore[no-any-return]
    return perspective.engine  # type: ignore[attr-defined]


def _try_extract_direct_points(state: GameState) -> Optional[tuple[int, int]]:
    """
    Best-effort extraction of (leader_points, follower_points).
    If your framework version exposes these differently, you can adjust this.
    """
    leader = state.leader
    follower = state.follower

    for score_attr in ("score", "points", "game_points"):
        if hasattr(leader, score_attr) and hasattr(follower, score_attr):
            ls = getattr(leader, score_attr)
            fs = getattr(follower, score_attr)
            for direct_attr in ("direct_points", "direct", "points", "value"):
                if hasattr(ls, direct_attr) and hasattr(fs, direct_attr):
                    try:
                        return int(getattr(ls, direct_attr)), int(getattr(fs, direct_attr))
                    except Exception:
                        pass
    return None


def _heuristic_value(state, maximizing: bool) -> float:
    leader = state.leader
    follower = state.follower

    # 1) Score (direct + discounted pending)
    pts = _try_extract_direct_points(state)
    if pts is None:
        base = 0.0
        l_pp = f_pp = 0
    else:
        l_dp, f_dp = pts
        l_pp = getattr(leader.score, "pending_points", 0)
        f_pp = getattr(follower.score, "pending_points", 0)
        base = (l_dp - f_dp) + 0.5 * (l_pp - f_pp)

    # 2) Trump count advantage
    trump = state.talon.trump_suit()
    leader_trumps = sum(1 for c in leader.hand.cards if c.suit == trump)
    follower_trumps = sum(1 for c in follower.hand.cards if c.suit == trump)
    trump_term = 1.0 * (leader_trumps - follower_trumps)

    # 3) Tempo (being leader is good)
    tempo_term = 0.5

    value = base + trump_term + tempo_term
    return value if maximizing else -value


@dataclass(frozen=True)
class SearchConfig:
    max_depth_phase1: int = 8
    use_heuristic: bool = True


class PerfectInfoBot(Bot):
    """
    Phase 1: perfect-info alpha-beta (cheats by peeking the full GameState)
    Phase 2: *regular* alpha-beta (no cheating), using the standard AlphaBetaBot logic
    """

    def __init__(self, name: Optional[str] = None, config: SearchConfig = SearchConfig()) -> None:
        super().__init__(name=name)
        self.config = config
        self._phase2_delegate = AlphaBetaBot(name=None)
        # rigging control
        self._rigged_once = False

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        if perspective.get_phase() == GamePhase.TWO:
            # Switch to regular phase-2 alpha-beta (no peeking)
            return self._phase2_delegate.get_move(perspective, leader_move)
        
        # Phase 1: perfect-info search using full state peek and cheat
        if not self._rigged_once and len(perspective.get_game_history()) == 0:
            _rig_to_all_trumps(perspective)
            self._rigged_once = True

        engine = _get_engine(perspective)
        state = _peek_full_state(perspective)

        _, move = self._value_phase1(
            state=state,
            engine=engine,
            leader_move=leader_move,
            maximizing=True,
            depth=self.config.max_depth_phase1,
            alpha=float("-inf"),
            beta=float("inf"),
        )
        return move

    def _value_phase1(
        self,
        state: GameState,
        engine: GamePlayEngine,
        leader_move: Optional[Move],
        maximizing: bool,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, Move]:
        # Use leader/follower perspectives to enumerate legal moves even in Phase 1
        if leader_move is None:
            my_perspective: PlayerPerspective = LeaderPerspective(state, engine)
        else:
            my_perspective = FollowerPerspective(state, engine, leader_move)

        valid_moves = my_perspective.valid_moves()
        # Phase 1 can include non-card leader moves (e.g., TrumpExchange).
        # Our recursion assumes "leader plays a card, follower responds", so filter to card-playing moves.
        if leader_move is None:
            valid_moves = [m for m in valid_moves if hasattr(m, "card")]

        assert valid_moves, "No valid moves found (unexpected)."

        best_value = float("-inf") if maximizing else float("inf")
        best_move: Optional[Move] = None

        for move in valid_moves:
            if leader_move is None:
                # choose leader move -> recurse to follower reply
                value, _ = self._value_phase1(
                    state=state,
                    engine=engine,
                    leader_move=move,
                    maximizing=not maximizing,
                    depth=depth,
                    alpha=alpha,
                    beta=beta,
                )
            else:
                # choose follower move -> complete trick -> recurse to next trick
                forced_leader = OneFixedMoveBot(leader_move)
                forced_follower = OneFixedMoveBot(move)

                new_state = engine.play_one_trick(
                    game_state=state,
                    new_leader=forced_leader,
                    new_follower=forced_follower,
                )

                winning_info = SchnapsenTrickScorer().declare_winner(new_state)
                if winning_info:
                    winner_impl = winning_info[0].implementation
                    points = float(winning_info[1])

                    follower_wins = (winner_impl == forced_follower)
                    if not follower_wins:
                        points = -points
                    if not maximizing:
                        points = -points
                    value = points
                else:
                    if depth <= 1:
                        value = _heuristic_value(new_state, maximizing) if self.config.use_heuristic else 0.0
                    else:
                        leader_stayed = (forced_leader == new_state.leader.implementation)
                        next_maximizing = (not maximizing) if leader_stayed else maximizing

                        value, _ = self._value_phase1(
                            state=new_state,
                            engine=engine,
                            leader_move=None,
                            maximizing=next_maximizing,
                            depth=depth - 1,
                            alpha=alpha,
                            beta=beta,
                        )

            # alpha-beta pruning
            if maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        assert best_move is not None
        return best_value, best_move

    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        # Reset so the next game can rig again
        self._rigged_once = False
