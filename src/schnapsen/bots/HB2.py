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
    """
    Plays exactly one predetermined move, but makes it legal in the CURRENT state copy
    by selecting the matching move from perspective.valid_moves().

    This is necessary because engine.play_one_trick() copies GameState, and Card objects
    (and sometimes Move instances) may not be identical across copies.
    """

    def __init__(self, move: Move) -> None:
        super().__init__(name=None)
        self._move: Optional[Move] = move

        # Store a "descriptor" for matching later
        self._cls = move.__class__
        self._has_card = hasattr(move, "card")
        if self._has_card:
            c = move.card  # type: ignore[attr-defined]
            self._rank = c.rank
            self._suit = c.suit
        else:
            self._rank = None
            self._suit = None

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        assert self._move is not None, "This bot can only play one move, after that it ends"

        # Consume it (one-shot bot)
        self._move = None

        # Choose a legal move from this *current* perspective that matches our descriptor
        candidates = perspective.valid_moves()

        # First: strict match by class + card identity by (rank,suit)
        if self._has_card:
            for m in candidates:
                if m.__class__ is self._cls and hasattr(m, "card"):
                    mc = m.card  # type: ignore[attr-defined]
                    if mc.rank == self._rank and mc.suit == self._suit:
                        return m

            # Second: looser match: any move with same card (rank,suit), regardless of class
            for m in candidates:
                if hasattr(m, "card"):
                    mc = m.card  # type: ignore[attr-defined]
                    if mc.rank == self._rank and mc.suit == self._suit:
                        return m

        # If we get here, we couldn't match it. Fallback to a legal move
        # rather than crashing the search engine.
        return candidates[0]


# =============================================================================
# Phase-1: PERFECT-INFO AlphaBeta (cheats by peeking full GameState)
# =============================================================================
def _rig_to_opening_hand(perspective: PlayerPerspective) -> None:
    """
    Research rig (Phase 1, at game start only):

    - Ensure the trump Jack is the face-up trump card at the bottom of the talon (talon_cards[-1]).
      Never modify talon_cards[-1] after that.
    - Force our hand to: A♠(trump), 10♠(trump), K♠(trump), Q♠(trump) + one Ace (prefer non-trump Ace).
    - Maintain invariants via 1-for-1 swaps only.
    """
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

    if not talon_cards:
        return  # shouldn't happen in phase 1

    def rank_name(c: Any) -> str:
        r = getattr(c, "rank", None)
        return getattr(r, "name", str(r))

    def is_trump(c: Any) -> bool:
        return getattr(c, "suit", None) == trump_suit

    # ----------------------------
    # Step 1: Move trump Jack to talon bottom and protect it
    # ----------------------------
    trump_jack = None
    # find trump jack in the whole pool
    for c in my_cards + opp_cards + talon_cards:
        if is_trump(c) and rank_name(c) == "JACK":
            trump_jack = c
            break
    if trump_jack is None:
        return  # unexpected, but don't corrupt state

    bottom_idx = len(talon_cards) - 1
    bottom_card = talon_cards[bottom_idx]

    if trump_jack is not bottom_card:
        # Put trump_jack at bottom, swap the displaced bottom_card back to where jack came from
        if trump_jack in talon_cards:
            j_idx = talon_cards.index(trump_jack)
            # swap within talon (safe because bottom_card is trump suit already)
            talon_cards[j_idx], talon_cards[bottom_idx] = talon_cards[bottom_idx], talon_cards[j_idx]
        elif trump_jack in my_cards:
            j_idx = my_cards.index(trump_jack)
            my_cards[j_idx] = bottom_card
            talon_cards[bottom_idx] = trump_jack
        elif trump_jack in opp_cards:
            j_idx = opp_cards.index(trump_jack)
            opp_cards[j_idx] = bottom_card
            talon_cards[bottom_idx] = trump_jack
        else:
            return

    protected = talon_cards[bottom_idx]  # this is now the trump jack
    # Never touch talon_cards[bottom_idx] again.

    # ----------------------------
    # Step 2: Build desired opening hand
    # ----------------------------
    pool = [c for c in (my_cards + opp_cards + talon_cards) if c is not protected]

    def find_card(pred) -> Optional[Any]:
        for c in pool:
            if pred(c):
                return c
        return None

    trump_ace = find_card(lambda c: is_trump(c) and rank_name(c) == "ACE")
    trump_ten = find_card(lambda c: is_trump(c) and rank_name(c) == "TEN")
    trump_king = find_card(lambda c: is_trump(c) and rank_name(c) == "KING")
    trump_queen = find_card(lambda c: is_trump(c) and rank_name(c) == "QUEEN")

    if None in (trump_ace, trump_ten, trump_king, trump_queen):
        return  # don't partially rig if something is weird

    # 5th card: an Ace, preferably non-trump
    non_trump_ace = find_card(lambda c: (not is_trump(c)) and rank_name(c) == "ACE")
    any_extra_ace = find_card(lambda c: rank_name(c) == "ACE" and c not in {trump_ace, trump_ten, trump_king, trump_queen})
    fifth = non_trump_ace or any_extra_ace
    if fifth is None:
        return

    desired = [trump_ace, trump_ten, trump_king, trump_queen, fifth]

    # If we already have exactly these 5 cards, just write back any prior swaps and exit
    if set(desired) == set(my_cards):
        me.hand.cards[:] = my_cards
        opp.hand.cards[:] = opp_cards
        state.talon._cards[:] = talon_cards  # type: ignore[attr-defined]
        return

    # ----------------------------
    # Step 3: 1-for-1 swap to make our hand exactly "desired"
    # ----------------------------
    missing = [c for c in desired if c not in my_cards]
    give_back = [c for c in my_cards if c not in desired]

    if len(missing) != len(give_back):
        return  # safety: avoid breaking invariants

    for take, give in zip(missing, give_back):
        if take in opp_cards:
            idx = opp_cards.index(take)
            opp_cards[idx] = give
        elif take in talon_cards:
            idx = talon_cards.index(take)
            # NEVER touch the protected bottom card
            if idx == bottom_idx:
                return
            talon_cards[idx] = give
        else:
            return

    # Set our hand to the exact desired cards
    my_cards = desired

    # ----------------------------
    # Step 4: Write back mutated lists (preserve invariants)
    # ----------------------------
    me.hand.cards[:] = my_cards
    opp.hand.cards[:] = opp_cards
    state.talon._cards[:] = talon_cards  # type: ignore[attr-defined]


# --- Card ordering: trump dominates, then rank strength (A > 10 > K > Q > J) ---
_RANK_STRENGTH = {
    "ACE": 5,
    "TEN": 4,
    "KING": 3,
    "QUEEN": 2,
    "JACK": 1,
}

def _rig_to_best_five(perspective: PlayerPerspective) -> None:
    state: GameState = perspective._PlayerPerspective__game_state  # type: ignore[attr-defined]
    trump_suit = state.talon.trump_suit()

    if perspective.am_i_leader():
        me = state.leader
        opp = state.follower
    else:
        me = state.follower
        opp = state.leader

    my_cards: List[Any] = list(me.hand.cards)
    opp_cards: List[Any] = list(opp.hand.cards)
    talon_cards: List[Any] = list(getattr(state.talon, "_cards", []))

    if not talon_cards:
        return
    protected = talon_cards[-1]               # face-up trump card (your trump jack)
    bottom_idx = len(talon_cards) - 1

    def rank_name(card: Any) -> str:
        r = getattr(card, "rank", None)
        return getattr(r, "name", str(r))

    def card_strength(card: Any) -> tuple[int, int]:
        is_trump = 1 if getattr(card, "suit", None) == trump_suit else 0
        return (is_trump, _RANK_STRENGTH.get(rank_name(card), 0))

    # Exclude protected bottom card from all selection
    pool = [c for c in (my_cards + opp_cards + talon_cards) if c is not protected]
    pool_sorted = sorted(pool, key=card_strength, reverse=True)

    best5 = pool_sorted[:5]

    # Absolute safety: ensure protected is not in best5, then refill to 5
    best5 = [c for c in best5 if c is not protected]
    if len(best5) < 5:
        for c in pool_sorted:
            if c is protected:
                continue
            if c not in best5:
                best5.append(c)
            if len(best5) == 5:
                break

    # If we already have exactly those 5 cards, do nothing
    if set(best5) == set(my_cards):
        return

    missing = [c for c in best5 if c not in my_cards]
    give_back = [c for c in my_cards if c not in best5]

    if len(missing) != len(give_back):
        return

    for take, give in zip(missing, give_back):
        if take in opp_cards:
            idx = opp_cards.index(take)
            opp_cards[idx] = give
        elif take in talon_cards:
            idx = talon_cards.index(take)
            if idx == bottom_idx:
                return  # never touch protected bottom card
            talon_cards[idx] = give
        else:
            return

    me.hand.cards[:] = best5
    opp.hand.cards[:] = opp_cards
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
        
        # Only rig when WE are the leader (leader_move is None).
        # Never mutate state after the opponent has already chosen a move.
        if leader_move is None:
            # First move of the game: force the opening hand exactly once
            if not self._rigged_once and len(perspective.get_game_history()) == 0:
                _rig_to_opening_hand(perspective)
                self._rigged_once = True
            else:
                # Subsequent tricks where we are leader: upgrade hand safely
                _rig_to_best_five(perspective)

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
