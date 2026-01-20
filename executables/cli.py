import random
import pathlib

from typing import Optional, List, Tuple, Any

import click
from schnapsen.alternative_engines.ace_one_engine import AceOneGamePlayEngine

from schnapsen.bots import MLDataBot, train_ML_model, MLPlayingBot, RandBot

from schnapsen.bots.example_bot import ExampleBot

from schnapsen.game import (Bot, GamePlayEngine, Move, PlayerPerspective,
                            SchnapsenGamePlayEngine, TrumpExchange)
from schnapsen.alternative_engines.twenty_four_card_schnapsen import TwentyFourSchnapsenGamePlayEngine

from schnapsen.bots.rdeep import RdeepBot

from schnapsen.bots.honest_bot import (
    HonestBot,
    SearchConfig,
)


@click.group()
def main() -> None:
    """Various Schnapsen Game Examples"""


def play_games_and_return_stats(engine: GamePlayEngine, bot1: Bot, bot2: Bot, pairs_of_games: int) -> int:
    """
    Play 2 * pairs_of_games games between bot1 and bot2, using the SchnapsenGamePlayEngine, and return how often bot1 won.
    Prints progress. Each pair of games is the same original dealing of cards, but the roles of the bots are swapped.
    """
    bot1_wins: int = 0
    lead, follower = bot1, bot2
    for game_pair in range(pairs_of_games):
        for lead, follower in [(bot1, bot2), (bot2, bot1)]:
            winner, _, _ = engine.play_game(lead, follower, random.Random(game_pair))
            if winner == bot1:
                bot1_wins += 1
        if game_pair > 0 and (game_pair + 1) % 500 == 0:
            print(f"Progress: {game_pair + 1}/{pairs_of_games} game pairs played")
    return bot1_wins

@main.command()
@click.option("--games", default=200, show_default=True, type=int)
@click.option("--seed0", default=1, show_default=True, type=int)
@click.option("--rdeep-samples", default=16, show_default=True, type=int)
@click.option("--rdeep-depth", default=4, show_default=True, type=int)
@click.option("--progress-every", default=5, show_default=True, type=int)
@click.option("--outfile", default="honest_vs_rdeep_report.txt", show_default=True, type=str)
def honest_vs_rdeep(
    games: int,
    seed0: int,
    rdeep_samples: int,
    rdeep_depth: int,
    progress_every: int,
    outfile: str,
) -> None:
    """
    HonestBot vs RdeepBot.
    - Alternates seats every game.
    - Fresh bots each game (no state leakage).
    - Clean progress printing (exactly once).
    - Full loss summaries written to outfile.
    """

    import time
    import random
    from typing import List, Tuple

    engine = SchnapsenGamePlayEngine()

    # Import your bot class (adjust path if needed)
    # from schnapsen.bots.honestbot import HonestBot
    # If HonestBot is defined in this same cli.py, then just use it directly.

    wins = 0
    losses: List[Tuple[int, str]] = []

    t0 = time.time()

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("HONEST VS RDEEP REPORT\n")
        f.write(f"games={games} seed0={seed0}\n")
        f.write(f"rdeep_samples={rdeep_samples} rdeep_depth={rdeep_depth}\n\n")

        for i in range(games):
            seed = seed0 + i
            rng = random.Random(seed)

            # Fresh bots each game
            honest = HonestBot(name="Honest")  # <-- your renamed bot
            rdeep = RdeepBot(
                num_samples=rdeep_samples,
                depth=rdeep_depth,
                rand=random.Random(99991 + seed),
            )

            # Alternate seats every game:
            # odd games: Honest is bot1 (leader at start)
            # even games: Honest is bot2
            if (i % 2) == 0:
                bot1, bot2 = honest, rdeep
                seats = "H-first"
            else:
                bot1, bot2 = rdeep, honest
                seats = "H-second"

            winner, game_points, score = engine.play_game(bot1, bot2, rng)

            honest_won = (winner == honest)
            if honest_won:
                wins += 1
            else:
                # Keep a compact loss line (don’t spam stdout)
                line = f"LOSS seed={seed} seats={seats} winner={winner} game_points={game_points} score={score}"
                losses.append((seed, line))
                f.write(line + "\n")

            # Progress printing EXACTLY ONCE
            if progress_every > 0 and ((i + 1) % progress_every == 0 or (i + 1) == games):
                print(f"Progress {i+1}/{games} | wins={wins} | winrate={wins/(i+1):.1%}")

        runtime = time.time() - t0
        f.write("\n")
        f.write(f"SUMMARY wins={wins}/{games} ({wins/games:.2%})\n")
        f.write(f"runtime_seconds={runtime:.2f}\n")
        f.write(f"losses={len(losses)}\n")
        if losses:
            f.write("loss_seeds:\n")
            for s, _ in losses:
                f.write(f"  - {s}\n")

    print(f"\nDONE. Wins={wins}/{games} ({wins/games:.2%})")
    print(f"Wrote report to: {outfile}")


@main.command()
@click.option("--games", default=1000, show_default=True, type=int, help="Games per opponent")
@click.option("--seed0", default=1, show_default=True, type=int, help="First seed used for random.Random(seed)")
@click.option("--outfile", default="honest_tournament_report.txt", show_default=True, type=str)
@click.option("--rdeep-samples", default=16, show_default=True, type=int)
@click.option("--rdeep-depth", default=4, show_default=True, type=int)
def honestbot_tournament(
    games: int,
    seed0: int,
    outfile: str,
    rdeep_samples: int,
    rdeep_depth: int,
) -> None:
    """
    Runs:
      - HonestBot vs RdeepBot (N games)
      - HonestBot vs BullyBot (N games)
      - HonestBot vs RandBot  (N games)

    Alternates seats every game by swapping bot1/bot2.
    Writes a report + loss summaries.
    """
    import time
    import random
    import inspect
    import statistics
    from typing import Optional, List, Dict, Any, Tuple

    from schnapsen.game import SchnapsenGamePlayEngine, Bot  # type: ignore

    # ---- your bot ----
    # If HonestBot is in the same file, remove this import.
    from schnapsen.bots.honestbot import HonestBot  # type: ignore

    engine = SchnapsenGamePlayEngine()

    # ----------------------------
    # Import helpers (repo-robust)
    # ----------------------------
    def _import_first(paths: List[str], name: str):
        last_err = None
        for p in paths:
            try:
                mod = __import__(p, fromlist=[name])
                return getattr(mod, name)
            except Exception as e:
                last_err = e
        raise last_err  # type: ignore[misc]

    # Try multiple common locations used in different schnapsen repos
    def _load_bots():
        RdeepBot = _import_first(
            [
                "schnapsen.bots.rdeep",
                "schnapsen.bots.rdeep_bot",
                "schnapsen.bots",
            ],
            "RdeepBot",
        )
        RandBot = _import_first(
            [
                "schnapsen.bots.rand",
                "schnapsen.bots.rand_bot",
                "schnapsen.bots",
            ],
            "RandBot",
        )
        BullyBot = _import_first(
            [
                "schnapsen.bots.bully_bot",   # <-- common in many repos
                "schnapsen.bots.bully",       # <-- your attempted path (fails in yours)
                "schnapsen.bots",
            ],
            "BullyBot",
        )
        return RdeepBot, BullyBot, RandBot

    # ----------------------------
    # Construction helpers
    # ----------------------------
    def _construct(bot_cls, seed: int, **preferred_kwargs):
        """
        Instantiate bot_cls, passing only kwargs that are accepted by its __init__.
        If it requires 'rand' or 'random', we provide random.Random(seed) automatically.
        """
        sig = None
        try:
            sig = inspect.signature(bot_cls.__init__)
        except Exception:
            sig = None

        kwargs = {}
        if sig is not None:
            params = sig.parameters

            # provide rand/random if requested
            if "rand" in params:
                kwargs["rand"] = random.Random(90000 + seed)
            if "random" in params:
                kwargs["random"] = random.Random(90000 + seed)

            # pass preferred kwargs only if accepted
            for k, v in preferred_kwargs.items():
                if k in params:
                    kwargs[k] = v

        # If signature unavailable, just try preferred_kwargs then fallback no-args
        try:
            return bot_cls(**kwargs)
        except TypeError:
            try:
                return bot_cls()
            except Exception:
                # last resort: try with preferred kwargs directly
                return bot_cls(**preferred_kwargs)

    # ----------------------------
    # Stats helpers
    # ----------------------------
    def _safe_int(x) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    def _extract_points(game_points) -> Optional[int]:
        if isinstance(game_points, int):
            return game_points
        if isinstance(game_points, float):
            return int(game_points)
        if isinstance(game_points, (tuple, list)):
            # sometimes returns (winner_points, loser_points) etc.
            for item in game_points:
                v = _safe_int(item)
                if v is not None:
                    return v
        return None

    def _extract_direct_points(score_obj) -> Optional[int]:
        # your log shows score like Score(direct_points=95, pending_points=0)
        if hasattr(score_obj, "direct_points"):
            v = _safe_int(getattr(score_obj, "direct_points"))
            if v is not None:
                return v
        return None

    def _run_block(opponent_name: str, opponent_factory, seed_start: int, n_games: int) -> Tuple[Dict[str, Any], int]:
        honest_wins = 0
        opp_wins = 0

        points_hist = {1: 0, 2: 0, 3: 0}
        honest_direct_pts_when_loss: List[int] = []
        opp_direct_pts_when_loss: List[int] = []

        loss_summaries: List[str] = []
        loss_seeds: List[int] = []

        t0 = time.time()

        for i in range(n_games):
            seed = seed_start + i

            honest = HonestBot(name="HonestBot")
            opp = opponent_factory(seed)

            bot1: Bot = honest
            bot2: Bot = opp
            if (i % 2) == 1:
                bot1, bot2 = bot2, bot1

            winner, game_points, score = engine.play_game(bot1, bot2, random.Random(seed))

            if winner == honest:
                honest_wins += 1
                p = _extract_points(game_points)
                if p in points_hist:
                    points_hist[p] += 1
            else:
                opp_wins += 1
                loss_seeds.append(seed)

                # Your output shows `score` is sometimes a Score object (not a tuple).
                # We at least capture direct_points if possible.
                dp = _extract_direct_points(score)
                if dp is not None:
                    # This dp appears to be "winner direct_points" in your printout.
                    opp_direct_pts_when_loss.append(dp)

                loss_summaries.append(
                    f"LOSS seed={seed} seats={'H-first' if bot1==honest else 'H-second'} "
                    f"game_points={game_points} score={score}"
                )

            if (i + 1) % max(1, n_games // 10) == 0:
                wr = honest_wins / (i + 1)
                print(f"[{opponent_name}] {i+1}/{n_games} | wins={honest_wins} | winrate={wr:.1%}")

        dt = time.time() - t0

        stats = {
            "opponent": opponent_name,
            "games": n_games,
            "wins": honest_wins,
            "losses": opp_wins,
            "winrate": honest_wins / n_games if n_games else 0.0,
            "points_hist_when_win": points_hist,
            "runtime_sec": dt,
            "loss_seeds": loss_seeds,
            "loss_summaries": loss_summaries,
            "opp_direct_points_when_loss_best_effort": (
                (statistics.mean(opp_direct_pts_when_loss), min(opp_direct_pts_when_loss), max(opp_direct_pts_when_loss))
                if opp_direct_pts_when_loss else None
            ),
        }
        return stats, seed_start + n_games

    # ----------------------------
    # Load opponent classes safely
    # ----------------------------
    try:
        RdeepBot, BullyBot, RandBot = _load_bots()
    except Exception as e:
        raise RuntimeError(f"Could not import one of the bots (Rdeep/Bully/Rand). Error: {e}")

    # ----------------------------
    # Opponent factories (seeded)
    # ----------------------------
    def make_rdeep(seed: int):
        # rdeep typically wants num_samples, depth, rand
        return _construct(
            RdeepBot,
            seed,
            num_samples=rdeep_samples,
            depth=rdeep_depth,
        )

    def make_bully(seed: int):
        # your error says BullyBot(rand=...) is required in YOUR repo
        return _construct(BullyBot, seed)

    def make_rand(seed: int):
        # may or may not want rand
        return _construct(RandBot, seed)

    blocks = [
        ("RdeepBot", make_rdeep),
        ("BullyBot", make_bully),
        ("RandBot", make_rand),
    ]

    # ----------------------------
    # Run tournament + write report
    # ----------------------------
    overall_games = 0
    overall_wins = 0
    overall_losses = 0

    seed_cursor = seed0

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("HONESTBOT TOURNAMENT REPORT\n")
        f.write(f"games_per_opponent={games} seed0={seed0}\n")
        f.write(f"rdeep_samples={rdeep_samples} rdeep_depth={rdeep_depth}\n\n")

        for name, factory in blocks:
            f.write("=" * 80 + "\n")
            f.write(f"{name}: {games} games (alternating seats)\n")
            f.write("=" * 80 + "\n")

            stats, seed_cursor = _run_block(name, factory, seed_cursor, games)

            overall_games += stats["games"]
            overall_wins += stats["wins"]
            overall_losses += stats["losses"]

            f.write(f"Wins: {stats['wins']}/{stats['games']} ({stats['winrate']:.2%})\n")
            f.write(f"Runtime: {stats['runtime_sec']:.2f}s\n")
            f.write(f"Points hist (when HonestBot wins): {stats['points_hist_when_win']}\n")
            f.write(f"Opponent direct_points on losses (mean/min/max, best-effort): {stats['opp_direct_points_when_loss_best_effort']}\n")

            if not stats["loss_seeds"]:
                f.write("Losses: 0 ✅\n\n")
            else:
                f.write(f"Losses: {len(stats['loss_seeds'])}\n")
                f.write("Loss seeds:\n")
                for s in stats["loss_seeds"]:
                    f.write(f"  - {s}\n")
                f.write("\nLoss summaries:\n")
                for line in stats["loss_summaries"]:
                    f.write("  " + line + "\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total: {overall_wins}/{overall_games} ({(overall_wins/overall_games):.2%})\n")
        f.write(f"Total losses: {overall_losses}\n")

    print("\nDONE.")
    print(f"Overall: {overall_wins}/{overall_games} ({(overall_wins/overall_games):.2%})")
    print(f"Wrote report to: {outfile}")


@main.command()
@click.option("--games", default=200, show_default=True, type=int, help="How many games to play")
@click.option("--seed0", default=1, show_default=True, type=int, help="First seed used for random.Random(seed)")
@click.option("--rdeep-samples", default=16, show_default=True, type=int)
@click.option("--rdeep-depth", default=4, show_default=True, type=int)
@click.option("--outfile", default="loss_report.txt", show_default=True, type=str)
def honest_diagnose(games: int, seed0: int, rdeep_samples: int, rdeep_depth: int, outfile: str) -> None:
    """
    HonestBot vs RdeepBot with LOSS logging:
    - Swaps seats every other game (like honest_vs_rdeep)
    - Reuses the SAME bot instances across all games (like a tournament)
    - Uses the same per-game RNG seeding style as honest_vs_rdeep (random.Random(game_number))
    - Only logs losses to outfile
    """

    engine = SchnapsenGamePlayEngine()

    # --- helper for readable move text ---
    def _move_to_str(m: Optional[Move]) -> str:
        if m is None:
            return "-"
        t = type(m).__name__
        if hasattr(m, "card"):
            c = m.card  # type: ignore[attr-defined]
            r = getattr(c.rank, "name", str(c.rank))
            s = getattr(c.suit, "name", str(c.suit))
            return f"{t}({r}_{s})"
        if hasattr(m, "jack"):
            j = m.jack  # type: ignore[attr-defined]
            r = getattr(j.rank, "name", str(j.rank))
            s = getattr(j.suit, "name", str(j.suit))
            return f"{t}(jack={r}_{s})"
        return t

    # --- Logging wrapper bot (same idea as before) ---
    class LoggingBot(Bot):
        """
        Wraps a Bot and records (phase, talon_len, leader_move?, chosen_move, scores if accessible).
        """
        def __init__(self, inner: Bot, label: str):
            super().__init__(name=f"{label}")
            self.inner = inner
            self.label = label
            self.log: List[str] = []

        def reset_log(self) -> None:
            self.log = []

        def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
            mv = self.inner.get_move(perspective, leader_move)

            # Gather cheap state info
            try:
                phase = perspective.get_phase()
            except Exception:
                phase = "?"

            try:
                st = perspective._PlayerPerspective__game_state  # type: ignore[attr-defined]
                talon_len = len(getattr(st.talon, "_cards", []))
            except Exception:
                talon_len = "?"

            # Try to read scores (best-effort)
            ldp = fdp = lpp = fpp = "?"
            try:
                st = perspective._PlayerPerspective__game_state  # type: ignore[attr-defined]
                ldp = getattr(st.leader.score, "direct_points", "?")
                fdp = getattr(st.follower.score, "direct_points", "?")
                lpp = getattr(st.leader.score, "pending_points", "?")
                fpp = getattr(st.follower.score, "pending_points", "?")
            except Exception:
                pass

            lm_str = _move_to_str(leader_move) if leader_move is not None else "-"
            mv_str = _move_to_str(mv)

            self.log.append(
                f"[{self.label}] phase={phase} talon={talon_len} "
                f"score(L:{ldp}+{lpp}, F:{fdp}+{fpp}) "
                f"leader_move={lm_str} -> move={mv_str}"
            )
            return mv

        def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
            if hasattr(self.inner, "notify_game_end"):
                self.inner.notify_game_end(won, perspective)

        def notify_trump_exchange(self, move) -> None:
            if hasattr(self.inner, "notify_trump_exchange"):
                self.inner.notify_trump_exchange(move)

    # ------------------------------------------------------------------
    # IMPORTANT: Tournament-style bot creation (ONE TIME, reused)
    # ------------------------------------------------------------------
    honest_inner = HonestBot(name="Honest")
    rdeep_inner = RdeepBot(
        num_samples=rdeep_samples,
        depth=rdeep_depth,
        rand=random.Random(4564654644),  # match honest_vs_rdeep style
    )

    honest = LoggingBot(honest_inner, "HB")
    rdeep = LoggingBot(rdeep_inner, "RD")

    bot1: Bot = honest
    bot2: Bot = rdeep

    wins = 0
    losses: List[int] = []

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("HonestBot vs RdeepBot — TOURNAMENT-STYLE LOSS DIAGNOSTICS\n\n")
        f.write(f"settings: games={games}, seed0={seed0}, rdeep_samples={rdeep_samples}, rdeep_depth={rdeep_depth}\n\n")

        for game_number in range(1, games + 1):
            # swap seats every other game (exactly like honest_vs_rdeep)
            if game_number % 2 == 0:
                bot1, bot2 = bot2, bot1

            # reset per-game logs
            honest.reset_log()
            rdeep.reset_log()

            # match the other command’s RNG style (random.Random(game_number))
            winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(game_number))

            # count wins for the *Honest* wrapper object
            if winner_id == honest:
                wins += 1
            else:
                losses.append(game_number)
                header = (
                    f"\n=== LOSS game_number={game_number} (seed={game_number}) "
                    f"bot1={type(bot1).__name__}:{getattr(bot1,'name', '')} "
                    f"bot2={type(bot2).__name__}:{getattr(bot2,'name', '')} "
                    f"winner={winner_id} points={game_points} score={score} ===\n"
                )
                f.write(header)
                f.write("\n-- Move log (PI) --\n")
                for line in honest.log:
                    f.write(line + "\n")
                f.write("\n-- Move log (RD) --\n")
                for line in rdeep.log:
                    f.write(line + "\n")

                print(f"LOSS at game {game_number}: points={game_points}, score={score}")

            # progress (every 5 games like your other function)
            if game_number % 5 == 0:
                print(
                    f"Honest won {wins} out of {game_number} games "
                    f"(last game: winner={winner_id}, points={game_points}, score={score})"
                )

        f.write(f"\nSUMMARY: wins={wins}/{games} ({wins/games:.1%})\n")
        if losses:
            f.write("\nLOSS GAME NUMBERS:\n")
            for g in losses:
                f.write(f"- {g}\n")

    print(f"\nDONE. Wins={wins}/{games} ({wins/games:.1%})")
    print(f"Wrote loss details to: {outfile}")


@main.command()
def random_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RandBot(random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


class NotificationExampleBot(Bot):

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        moves = perspective.valid_moves()
        return moves[0]

    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        print(f'result {"win" if won else "lost"}')
        print(f'I still have {len(perspective.get_hand())} cards left')

    def notify_trump_exchange(self, move: TrumpExchange) -> None:
        print(f"That trump exchanged! {move.jack}")


@main.command()
def notification_game() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = NotificationExampleBot()
    bot2 = RandBot(random.Random(464566))
    engine.play_game(bot1, bot2, random.Random(94))


class HistoryBot(Bot):
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        history = perspective.get_game_history()
        print(f'the initial state of this game was {history[0][0]}')
        moves = perspective.valid_moves()
        return moves[0]


@main.command()
def try_example_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = ExampleBot()
    bot2 = RandBot(random.Random(464566))
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(f"Winner is: {winner}, with {points} points, score {score}!")


@main.command()
def rdeep_game() -> None:
    bot1: Bot
    bot2: Bot
    engine = SchnapsenGamePlayEngine()
    rdeep = bot1 = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))
    bot2 = RandBot(random.Random(464566))
    wins = 0
    amount = 100
    for game_number in range(1, amount + 1):
        if game_number % 2 == 0:
            bot1, bot2 = bot2, bot1
        winner_id, _, _ = engine.play_game(bot1, bot2, random.Random(game_number))
        if winner_id == rdeep:
            wins += 1
        if game_number % 10 == 0:
            print(f"won {wins} out of {game_number}")


@main.group()
def ml() -> None:
    """Commands for the ML bot"""


@ml.command()
def create_replay_memory_dataset() -> None:
    # define replay memory database creation parameters
    num_of_games: int = 10000
    replay_memory_dir: str = 'ML_replay_memories'
    replay_memory_filename: str = 'random_random_10k_games.txt'
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename

    bot_1_behaviour: Bot = RandBot(random.Random(5234243))
    # bot_1_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(4564654644))
    bot_2_behaviour: Bot = RandBot(random.Random(54354))
    # bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(68438))
    delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
    if delete_existing_older_dataset and replay_memory_location.exists():
        print(f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected.")
        replay_memory_location.unlink()

    # in any case make sure the directory exists
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

    # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_location=replay_memory_location)
    replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_location=replay_memory_location)
    for i in range(1, num_of_games + 1):
        if i % 500 == 0:
            print(f"Progress: {i}/{num_of_games}")
        engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(i))
    print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}")


@ml.command()
def train_model() -> None:
    # directory where the replay memory is saved
    replay_memory_filename: str = 'random_random_10k_games.txt'
    # filename of replay memory within that directory
    replay_memories_directory: str = 'ML_replay_memories'
    # Whether to train a complicated Neural Network model or a simple one.
    # Tips: a neural network usually requires bigger datasets to be trained on, and to play with the parameters of the model.
    # Feel free to play with the hyperparameters of the model in file 'ml_bot.py', function 'train_ML_model',
    # under the code of body of the if statement 'if use_neural_network:'
    replay_memory_location = pathlib.Path(replay_memories_directory) / replay_memory_filename
    model_name: str = 'simple_model'
    model_dir: str = "ML_models"
    model_location = pathlib.Path(model_dir) / model_name
    overwrite: bool = False

    if overwrite and model_location.exists():
        print(f"Model at {model_location} exists already and will be overwritten as selected.")
        model_location.unlink()

    train_ML_model(replay_memory_location=replay_memory_location, model_location=model_location,
                   model_class='LR')


@ml.command()
def try_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    model_dir: str = 'ML_models'
    model_name: str = 'simple_model'
    model_location = pathlib.Path(model_dir) / model_name
    bot1: Bot = MLPlayingBot(model_location=model_location)
    bot2: Bot = RandBot(random.Random(464566))
    number_of_games: int = 10000
    pairs_of_games = number_of_games // 2

    # play games with altering leader position on first rounds
    ml_bot_wins_against_random = play_games_and_return_stats(engine=engine, bot1=bot1, bot2=bot2, pairs_of_games=pairs_of_games)
    print(f"The ML bot with name {model_name}, won {ml_bot_wins_against_random} times out of {number_of_games} games played.")


@main.command()
def game_24() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RandBot(random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


@main.command()
def game_ace_one() -> None:
    engine = AceOneGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RdeepBot(num_samples=16, depth=4, rand=random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


if __name__ == "__main__":
    main()
