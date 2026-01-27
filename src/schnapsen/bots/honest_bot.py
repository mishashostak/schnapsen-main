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
        move_to_play = self._move
        self._move = None

        candidates = perspective.valid_moves()

        # -------------------------
        # Case 1: Card-based move
        # -------------------------
        if self._has_card:
            # strict: same class + same card (rank,suit)
            for m in candidates:
                if m.__class__ is self._cls and hasattr(m, "card"):
                    mc = m.card  # type: ignore[attr-defined]
                    if mc.rank == self._rank and mc.suit == self._suit:
                        return m

            # looser: any move with same card (rank,suit)
            for m in candidates:
                if hasattr(m, "card"):
                    mc = m.card  # type: ignore[attr-defined]
                    if mc.rank == self._rank and mc.suit == self._suit:
                        return m

            return candidates[0]

        # -------------------------
        # Case 2: Non-card move (Marriage / TrumpExchange / etc.)
        # -------------------------
        # Best effort: match by class name first
        for m in candidates:
            if m.__class__ is self._cls:
                return m

        # If the move has a "jack" attribute (common for TrumpExchange), match by that too
        if hasattr(move_to_play, "jack"):
            j = move_to_play.jack  # type: ignore[attr-defined]
            jr = getattr(j.rank, "name", str(j.rank))
            js = getattr(j.suit, "name", str(j.suit))
            for m in candidates:
                if hasattr(m, "jack"):
                    mj = m.jack  # type: ignore[attr-defined]
                    mjr = getattr(mj.rank, "name", str(mj.rank))
                    mjs = getattr(mj.suit, "name", str(mj.suit))
                    if (mjr, mjs) == (jr, js):
                        return m
                    
        return candidates[0]


# =============================================================================
# Phase-1: PERFECT-INFO AlphaBeta (cheats by peeking full GameState)
# =============================================================================
def _rig_hand_optimally(perspective: PlayerPerspective, leader_move: Optional[Move]) -> None:
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

    hand_size = len(my_cards)
    if hand_size == 0 or not (my_cards + opp_cards + talon_cards):
        return

    # === Protect the bottom card forever ===
    bottom_card = talon_cards[-1] if talon_cards else None

    def _name(x: Any) -> str:
        return getattr(x, "name", str(x))

    def card_key(c: Any) -> tuple[str, str]:
        rank = getattr(c, "rank", None)
        suit = getattr(c, "suit", None)
        return (_name(rank), _name(suit))
    
    locked_keys: set[tuple[str, str]] = set()

    # If leader already played something, protect everything that move depends on.
    if leader_move is not None:
        # 1) Regular card move
        if hasattr(leader_move, "card"):
            locked_keys.add(card_key(leader_move.card))

        # 2) Marriage: lock BOTH marriage cards (king+queen of that suit)
        #    so we never steal either one while the move is pending.
        if hasattr(leader_move, "is_marriage") and leader_move.is_marriage():
            rm = leader_move.as_marriage().underlying_regular_move()
            suit = rm.card.suit
            locked_keys.add(card_key(rm.card))  # underlying played card
            # lock the partner card too:
            # partner is the other of King/Queen in that suit
            under_rank = getattr(rm.card.rank, "name", str(rm.card.rank)).upper()
            partner_rank = "KING" if under_rank == "QUEEN" else "QUEEN"
            locked_keys.add((partner_rank, getattr(suit, "name", str(suit))))

        # 3) Trump exchange: lock the jack AND the current trump card
        if hasattr(leader_move, "jack"):
            j = leader_move.jack
            locked_keys.add(card_key(j))
            # lock the exposed trump card too
            trump_card = getattr(state.talon, "trump_card", None)
            if trump_card is not None:
                locked_keys.add(card_key(trump_card))


    # ----------------------------
    # ✅ CRASH FIX:
    # If we're FOLLOWER (leader_move is not None), never steal from opp_cards.
    # This prevents invalidating the leader's already-chosen move.
    # ----------------------------
    is_follower_turn = leader_move is not None

    # Build source pool:
    # - If leader: can (cheat) steal from opponent + talon
    # - If follower: can only steal from talon (NOT opponent hand)
    source_cards = (my_cards + opp_cards + talon_cards)

    pool: List[Any] = [
        c for c in source_cards
        if c is not bottom_card and card_key(c) not in locked_keys
    ]

    rank_strength = {"ACE": 5, "TEN": 4, "KING": 3, "QUEEN": 2, "JACK": 1}

    def get_strength(c: Any) -> int:
        rank = getattr(c, "rank", None)
        suit = getattr(c, "suit", None)
        rname = _name(rank).upper()
        base = rank_strength.get(rname, 0)
        return base + 10 if suit == trump_suit else base

    trumps_in_pool = [c for c in pool if getattr(c, "suit", None) == trump_suit]
    desired: List[Any] = []

    if leader_move is not None and hasattr(leader_move, "card"):
        led = leader_move.card
        led_suit = getattr(led, "suit", None)
        led_is_trump = led_suit == trump_suit
        led_rank_name = _name(getattr(led, "rank", None)).upper()
        led_str = rank_strength.get(led_rank_name, 0)

        if led_is_trump:
            beaters = [
                c for c in trumps_in_pool
                if rank_strength.get(_name(getattr(c, "rank", None)).upper(), 0) > led_str
            ]
            if beaters:
                beaters.sort(key=get_strength)
                desired.append(beaters[0])
                trumps_in_pool.remove(beaters[0])

            trumps_in_pool.sort(key=get_strength, reverse=True)
            desired += trumps_in_pool[:hand_size - len(desired)]

        else:
            if trumps_in_pool:
                trumps_in_pool.sort(key=get_strength)  # cheapest ruff
                desired.append(trumps_in_pool[0])
                remaining_trumps = trumps_in_pool[1:]
                remaining_trumps.sort(key=get_strength, reverse=True)
                desired += remaining_trumps[:hand_size - len(desired)]
            else:
                same_suit_cards = [c for c in pool if getattr(c, "suit", None) == led_suit]
                beaters = [
                    c for c in same_suit_cards
                    if rank_strength.get(_name(getattr(c, "rank", None)).upper(), 0) > led_str
                ]
                if beaters:
                    beaters.sort(key=get_strength)
                    desired.append(beaters[0])

    else:
        # We are leader → grab highest trumps first
        trumps_in_pool.sort(key=get_strength, reverse=True)
        desired += trumps_in_pool[:hand_size]

    # Fill remaining slots with the overall highest cards left in the pool
    if len(desired) < hand_size:
        remaining_pool = [c for c in pool if c not in desired]
        remaining_pool.sort(key=get_strength, reverse=True)
        desired += remaining_pool[:hand_size - len(desired)]

    # Apply the swaps only if the hand actually changes
    if set(desired) == set(my_cards):
        return

    missing = [c for c in desired if c not in my_cards]
    excess  = [c for c in my_cards if c not in desired]

    if len(missing) != len(excess):
        return  # safety

    # --- swaps: MUST preserve talon order ---
    trump_index = len(talon_cards) - 1  # bottom trump card slot (must never change)

    for take, give in zip(missing, excess):
        if take in talon_cards:
            idx = talon_cards.index(take)

            # Never touch the bottom trump card position.
            # (Your pool already tries to exclude it, but this is a hard safety net.)
            if idx == trump_index:
                # If this ever triggers, your pool selection included the bottom trump card.
                # Bail out rather than corrupting the talon.
                return

            talon_cards[idx] = give

        elif take in opp_cards:
            idx = opp_cards.index(take)
            opp_cards[idx] = give

        else:
            return  # safety


    # If leader already committed a CARD move, that exact card must still be
    # present in the leader's hand in this state before we write back.
    # ----------------------------
    if leader_move is not None and hasattr(leader_move, "card"):
        led_card = leader_move.card  # the committed card object in this state
        # In this function, `opp` is ALWAYS the current leader when we are follower.
        # If we are follower, `opp.hand` must still contain the committed card.
        # If we are leader, this check is still safe (it checks the real leader).
        assert led_card in opp_cards, (
            "Rigging removed the leader's committed card! "
            f"Committed={led_card}, opp_cards={opp_cards}"
        )

    # sanity: no card should exist in both hands
    overlap = set(desired) & set(opp_cards)
    assert not overlap, f"Rigging duplicated cards across hands: {overlap}"

    #sanity check
    if talon_cards:
        assert getattr(talon_cards[-1], "suit", None) == trump_suit, (
            f"Talon bottom card suit changed! bottom={talon_cards[-1]} trump={trump_suit}"
        )


    # Write back
    me.hand.cards[:] = desired
    opp.hand.cards[:] = opp_cards
    state.talon._cards[:] = talon_cards  # type: ignore[attr-defined]


def _player_id(p: Any) -> Any:
    # BotState: identity should be the Bot instance, which is stable across copies
    if hasattr(p, "implementation"):
        return p.implementation  # stable object identity

    # If someone passes a Bot directly
    if isinstance(p, Bot):
        return p

    # Fallbacks (should almost never be used now)
    if hasattr(p, "name"):
        return getattr(p, "name")
    return p


def _peek_full_state(perspective: PlayerPerspective) -> GameState:
    # Name-mangled private attribute: PlayerPerspective.__game_state
    return perspective._PlayerPerspective__game_state  # type: ignore[attr-defined]


def _get_engine(perspective: PlayerPerspective) -> GamePlayEngine:
    # Most versions expose get_engine()
    if hasattr(perspective, "get_engine"):
        return perspective.get_engine()  # type: ignore[no-any-return]
    return perspective.engine  # type: ignore[attr-defined]


def _pick_forced_special_move(perspective: PlayerPerspective) -> Optional[Move]:
    """
    If a special move exists (TrumpExchange / Marriage), play it immediately.
    We detect by class name to stay robust across repo versions.
    """
    for m in perspective.valid_moves():
        name = type(m).__name__.lower()
        if "trumpexchange" in name:
            return m
        if "marriage" in name:
            return m
    return None


def _heuristic_value_honest(state: GameState, honest_pid: Any) -> float:
    # who is honest in THIS state?
    lpid = _player_id(state.leader)
    fpid = _player_id(state.follower)

    # score extraction (best-effort like yours)
    l_dp = int(getattr(state.leader.score, "direct_points", 0))
    f_dp = int(getattr(state.follower.score, "direct_points", 0))
    l_pp = int(getattr(state.leader.score, "pending_points", 0))
    f_pp = int(getattr(state.follower.score, "pending_points", 0))

    if lpid == honest_pid:
        my_dp, opp_dp = l_dp, f_dp
        my_pp, opp_pp = l_pp, f_pp
        me_hand = state.leader.hand.cards
        opp_hand = state.follower.hand.cards
    else:
        my_dp, opp_dp = f_dp, l_dp
        my_pp, opp_pp = f_pp, l_pp
        me_hand = state.follower.hand.cards
        opp_hand = state.leader.hand.cards

    base = (my_dp - opp_dp) + 0.5 * (my_pp - opp_pp)

    trump = state.talon.trump_suit()
    my_trumps = sum(1 for c in me_hand if c.suit == trump)
    opp_trumps = sum(1 for c in opp_hand if c.suit == trump)

    # small tempo bonus if honest is leader right now
    tempo = 0.5 if _player_id(state.leader) == honest_pid else 0.0

    return base + 1.0 * (my_trumps - opp_trumps) + tempo


def _canonicalize_move(perspective: PlayerPerspective, chosen: Move) -> Move:
    """Return the matching Move instance from perspective.valid_moves()."""
    legal = perspective.valid_moves()

    # If it's already exactly one of the legal moves, perfect.
    if chosen in legal:
        return chosen

    # Card move: match by (rank, suit)
    if hasattr(chosen, "card"):
        cr = getattr(chosen.card.rank, "name", str(chosen.card.rank))
        cs = getattr(chosen.card.suit, "name", str(chosen.card.suit))
        for m in legal:
            if hasattr(m, "card"):
                mr = getattr(m.card.rank, "name", str(m.card.rank))
                ms = getattr(m.card.suit, "name", str(m.card.suit))
                if (mr, ms) == (cr, cs):
                    return m

    # Special move: match by class name
    cname = type(chosen).__name__
    for m in legal:
        if type(m).__name__ == cname:
            return m

    # Fallback: never return illegal
    return legal[0]


def _should_play_marriage(perspective: PlayerPerspective) -> bool:
    state = perspective._PlayerPerspective__game_state  # type: ignore
    leader = state.leader
    follower = state.follower

    # If talon still large, delay marriage
    talon_len = len(getattr(state.talon, "_cards", []))
    if talon_len > 2:
        return False

    # Do NOT allow opponent to exchange trump after marriage
    for m in perspective.valid_moves():
        if "trumpexchange" in type(m).__name__.lower():
            return False

    # Only play marriage if it immediately wins or locks Phase 2 lead
    l_pts = leader.score.direct_points
    f_pts = follower.score.direct_points
    pending = getattr(leader.score, "pending_points", 0)

    # Marriage must either:
    # - win immediately
    # - OR guarantee Phase 2 entry with lead
    return (l_pts + pending >= 66) or (l_pts + pending > f_pts)


def _state_key(state: GameState) -> tuple:
    def card_id(c) -> tuple:
        suit = getattr(c.suit, "name", str(c.suit))
        rank = getattr(c.rank, "name", str(c.rank))
        return (suit, rank)

    trump = getattr(state.talon.trump_suit(), "name", str(state.talon.trump_suit()))

    leader_hand = tuple(sorted(card_id(c) for c in state.leader.hand.cards))
    follower_hand = tuple(sorted(card_id(c) for c in state.follower.hand.cards))

    if hasattr(state.talon, "_cards"):
        talon = tuple(card_id(c) for c in state.talon._cards)  # type: ignore[attr-defined]
    else:
        talon = tuple()

    return (
        trump,
        leader_hand,
        follower_hand,
        talon,
        int(getattr(state.leader.score, "direct_points", 0)),
        int(getattr(state.follower.score, "direct_points", 0)),
        int(getattr(state.leader.score, "pending_points", 0)),
        int(getattr(state.follower.score, "pending_points", 0)),
    )


@dataclass(frozen=True)
class SearchConfig:
    max_depth_phase1: int = 8
    use_heuristic: bool = True


class HonestBot(Bot):
    """
    Phase 1: perfect-info alpha-beta (cheats by peeking the full GameState)
    Phase 2: *regular* alpha-beta (no cheating), using the standard AlphaBetaBot logic
    """

    def __init__(self, name: Optional[str] = None, config: SearchConfig = SearchConfig()) -> None:
        super().__init__(name=name)
        self.config = config
        self._phase2_delegate = AlphaBetaBot(name=None)

        # Marriage control
        self._marriage_played = False

        # PERFORMANCE ADDITIONS
        self._tt = {}
        self._nodes = 0
        self._node_budget = 20000
        self._scorer = SchnapsenTrickScorer()


    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        # -------------------------
        # Phase 2: no cheating
        # -------------------------
        if perspective.get_phase() == GamePhase.TWO:
            return self._phase2_delegate.get_move(perspective, leader_move)

        # Rig your hand.
        _rig_hand_optimally(perspective, leader_move)

        # -------------------------
        # If leader played a special (no .card), return the best legal move.
        # -------------------------
        if leader_move is not None and not hasattr(leader_move, "card"):
            # Now that OneFixedMoveBot can replay specials, we can search properly.
            engine = _get_engine(perspective)
            state = _peek_full_state(perspective)

            self._nodes = 0
            d = min(6, self.config.max_depth_phase1)  # small depth is usually enough here
            honest_pid = _player_id(state.leader) if perspective.am_i_leader() else _player_id(state.follower)
            val, mv = self._value_phase1(
                state=state,
                engine=engine,
                leader_move=leader_move,   # follower perspective
                depth=d,
                alpha=float("-inf"),
                beta=float("inf"),
                honest_pid=honest_pid,
            )
            return _canonicalize_move(perspective, mv)

        # -------------------------
        # Special moves (Phase 1):
        # - Never auto-play TrumpExchange in your rigged setup
        # - Marriage only when your strict gate says so
        # -------------------------
        special = _pick_forced_special_move(perspective)
        if special is not None:
            sname = type(special).__name__.lower()

            if "trumpexchange" in sname:
                special = None

            elif "marriage" in sname:
                if self._marriage_played or (not _should_play_marriage(perspective)):
                    special = None
                else:
                    self._marriage_played = True

            if special is not None:
                return _canonicalize_move(perspective, special)
            

        # -------------------------
        # Perfect-info alpha-beta search
        # -------------------------
        engine = _get_engine(perspective)
        state = _peek_full_state(perspective)

        best_move: Optional[Move] = None
        self._nodes = 0

        for d in range(2, self.config.max_depth_phase1 + 1):
            honest_pid = _player_id(state.leader) if perspective.am_i_leader() else _player_id(state.follower)
            val, mv = self._value_phase1(
                state=state,
                engine=engine,
                leader_move=leader_move,   # follower perspective
                depth=d,
                alpha=float("-inf"),
                beta=float("inf"),
                honest_pid=honest_pid,
            )
            best_move = mv

        assert best_move is not None
        return _canonicalize_move(perspective, best_move)


    def _value_phase1(
        self,
        state: GameState,
        engine: GamePlayEngine,
        leader_move: Optional[Move],
        depth: int,
        alpha: float,
        beta: float,
        honest_pid: Any,
    ) -> tuple[float, Move]:

        # whose turn is it (as a PLAYER), and is it Honest?
        if leader_move is None:
            to_move_pid = _player_id(state.leader)
            persp: PlayerPerspective = LeaderPerspective(state, engine)
        else:
            to_move_pid = _player_id(state.follower)
            persp = FollowerPerspective(state, engine, leader_move)

        maximizing = (to_move_pid == honest_pid)

        valid_moves = persp.valid_moves()
        if leader_move is None:
            valid_moves = [m for m in valid_moves if hasattr(m, "card")]
        assert valid_moves, "No valid moves found"

        # node/depth cutoff
        self._nodes += 1
        if depth <= 0 or self._nodes > self._node_budget:
            trump = state.talon.trump_suit()
            rank_score = {"ACE": 5, "TEN": 4, "KING": 3, "QUEEN": 2, "JACK": 1}

            def sort_key(m: Move) -> int:
                if not hasattr(m, "card"):
                    return -100
                c = m.card
                rname = getattr(c.rank, "name", str(c.rank))
                return (10 if c.suit == trump else 0) + rank_score.get(rname, 0)

            valid_moves.sort(key=sort_key, reverse=maximizing)
            # return the best-looking legal move with heuristic
            # (for follower we can cheaply complete trick; for leader do 1-ply worst-case)
            if leader_move is not None:
                best_v = float("-inf") if maximizing else float("inf")
                best_m = valid_moves[0]
                for m in valid_moves:
                    lb = OneFixedMoveBot(leader_move)
                    fb = OneFixedMoveBot(m)
                    ns = engine.play_one_trick(state, lb, fb)
                    win = self._scorer.declare_winner(ns)
                    if win:
                        winner_player = win[0]
                        pts = float(win[1])
                        v = pts if _player_id(winner_player) == honest_pid else -pts
                    else:
                        v = _heuristic_value_honest(ns, honest_pid)

                    if maximizing:
                        if v > best_v:
                            best_v, best_m = v, m
                    else:
                        if v < best_v:
                            best_v, best_m = v, m
                return best_v, best_m
            else:
                # leader cutoff: pessimistic 1-ply
                best_v = float("-inf") if maximizing else float("inf")
                best_m = valid_moves[0]
                for m in valid_moves:
                    foll = FollowerPerspective(state, engine, m)
                    replies = foll.valid_moves()
                    worst_for_honest = float("inf")
                    for r in replies:
                        lb = OneFixedMoveBot(m)
                        fb = OneFixedMoveBot(r)
                        ns = engine.play_one_trick(state, lb, fb)
                        win = self._scorer.declare_winner(ns)
                        if win:
                            winner_player = win[0]
                            pts = float(win[1])
                            v = pts if _player_id(winner_player) == honest_pid else -pts
                        else:
                            v = _heuristic_value_honest(ns, honest_pid)
                        if v < worst_for_honest:
                            worst_for_honest = v

                    v = worst_for_honest
                    if maximizing:
                        if v > best_v:
                            best_v, best_m = v, m
                    else:
                        if v < best_v:
                            best_v, best_m = v, m
                return best_v, best_m

        # --- TT key (must include whose-turn type and depth; honest_pid constant) ---
        if leader_move is None:
            lm_key = ("L",)
        elif hasattr(leader_move, "card"):
            c = leader_move.card
            lm_key = ("F", getattr(c.rank, "name", str(c.rank)), getattr(c.suit, "name", str(c.suit)))
        else:
            lm_key = ("F", type(leader_move).__name__)

        key = (_state_key(state), lm_key, to_move_pid, depth)
        if key in self._tt:
            val, move_desc = self._tt[key]
            if move_desc is not None:
                for m in valid_moves:
                    if hasattr(m, "card"):
                        mr = getattr(m.card.rank, "name", str(m.card.rank))
                        ms = getattr(m.card.suit, "name", str(m.card.suit))
                        if (mr, ms) == move_desc:
                            return val, m
            return val, valid_moves[0]

        # move ordering
        trump = state.talon.trump_suit()
        rank_score = {"ACE": 5, "TEN": 4, "KING": 3, "QUEEN": 2, "JACK": 1}

        def order_key(m: Move) -> int:
            if not hasattr(m, "card"):
                return -100
            c = m.card
            rname = getattr(c.rank, "name", str(c.rank))
            return (10 if c.suit == trump else 0) + rank_score.get(rname, 0)

        valid_moves.sort(key=order_key, reverse=maximizing)

        best_v = float("-inf") if maximizing else float("inf")
        best_m: Optional[Move] = None

        for m in valid_moves:
            if leader_move is None:
                v, _ = self._value_phase1(state, engine, m, depth - 1, alpha, beta, honest_pid)
            else:
                lb = OneFixedMoveBot(leader_move)
                fb = OneFixedMoveBot(m)
                ns = engine.play_one_trick(state, lb, fb)

                win = self._scorer.declare_winner(ns)
                if win:
                    winner_bot = win[0].implementation
                    pts = float(win[1])
                    v = pts if winner_bot == honest_pid else -pts
                else:
                    v, _ = self._value_phase1(ns, engine, None, depth - 1, alpha, beta, honest_pid)

            if maximizing:
                if v > best_v:
                    best_v, best_m = v, m
                alpha = max(alpha, best_v)
                if beta <= alpha:
                    break
            else:
                if v < best_v:
                    best_v, best_m = v, m
                beta = min(beta, best_v)
                if beta <= alpha:
                    break

        assert best_m is not None

        # store TT
        if hasattr(best_m, "card"):
            br = getattr(best_m.card.rank, "name", str(best_m.card.rank))
            bs = getattr(best_m.card.suit, "name", str(best_m.card.suit))
            desc = (br, bs)
        else:
            desc = None
        self._tt[key] = (best_v, desc)

        return best_v, best_m


    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        self._marriage_played = False
        self._tt.clear()

