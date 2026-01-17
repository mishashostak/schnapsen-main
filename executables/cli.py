import random
import pathlib

from typing import Optional, List, Tuple

import click
from schnapsen.alternative_engines.ace_one_engine import AceOneGamePlayEngine

from schnapsen.bots import MLDataBot, train_ML_model, MLPlayingBot, RandBot

from schnapsen.bots.example_bot import ExampleBot

from schnapsen.game import (Bot, GamePlayEngine, Move, PlayerPerspective,
                            SchnapsenGamePlayEngine, TrumpExchange)
from schnapsen.alternative_engines.twenty_four_card_schnapsen import TwentyFourSchnapsenGamePlayEngine

from schnapsen.bots.rdeep import RdeepBot

from schnapsen.bots.perfectinfo import (
    PerfectInfoBot,
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
def perfectinfo_vs_rdeep() -> None:
    """
    Play your perfect-info Phase1 AlphaBeta + Phase2 regular AlphaBeta bot vs RdeepBot.
    Swaps seats every other game, similar to rdeep_game().
    """
    engine = SchnapsenGamePlayEngine()

    # Your perfect-info bot (tune depth as you like)
    perfect = PerfectInfoBot(
        name="PerfectInfo",
        config=SearchConfig(max_depth_phase1= 7, use_heuristic=True), #CHANGE MAX DEPTH HERE
    )

    # Rdeep bot (same params style as your rdeep_game)
    rdeep = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))

    bot1: Bot = perfect
    bot2: Bot = rdeep

    wins = 0
    amount = 200

    for game_number in range(1, amount + 1):
        # swap roles every other game
        if game_number % 2 == 0:
            bot1, bot2 = bot2, bot1

        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(game_number))

        # count wins for the perfect-info bot
        if winner_id == perfect:
            wins += 1

        if game_number % 5 == 0:
            print(f"PerfectInfo won {wins} out of {game_number} games "
                  f"(last game: winner={winner_id}, points={game_points}, score={score})")


@main.command()
@click.option("--games", default=200, show_default=True, type=int, help="How many games to play")
@click.option("--seed0", default=1, show_default=True, type=int, help="First seed used for random.Random(seed)")
@click.option("--rdeep-samples", default=16, show_default=True, type=int)
@click.option("--rdeep-depth", default=4, show_default=True, type=int)
@click.option("--outfile", default="loss_report.txt", show_default=True, type=str)
def perfectinfo_diagnose(games: int, seed0: int, rdeep_samples: int, rdeep_depth: int, outfile: str) -> None:
    """
    Run PerfectInfoBot vs RdeepBot for many games.
    For each LOSS, dump a compact, readable replay to a text file you can share.
    """

    engine = SchnapsenGamePlayEngine()

    # --- Logging wrapper bot ---
    class LoggingBot:
        """
        Wraps a Bot and records (phase, talon_len, leader_move?, chosen_move, scores if accessible).
        """
        def __init__(self, inner, label: str):
            self.inner = inner
            self.label = label
            self.log: List[str] = []

        def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
            # Ask inner bot
            mv = self.inner.get_move(perspective, leader_move)

            # Gather cheap state info
            try:
                phase = perspective.get_phase()
            except Exception:
                phase = "?"

            try:
                talon_len = len(getattr(perspective._PlayerPerspective__game_state.talon, "_cards", []))  # type: ignore[attr-defined]
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
            # Forward if inner has it
            if hasattr(self.inner, "notify_game_end"):
                self.inner.notify_game_end(won, perspective)

        def notify_trump_exchange(self, move) -> None:
            if hasattr(self.inner, "notify_trump_exchange"):
                self.inner.notify_trump_exchange(move)

    def _move_to_str(m: Optional[Move]) -> str:
        if m is None:
            return "-"
        t = type(m).__name__
        if hasattr(m, "card"):
            c = m.card  # type: ignore[attr-defined]
            r = getattr(c.rank, "name", str(c.rank))
            s = getattr(c.suit, "name", str(c.suit))
            return f"{t}({r}_{s})"
        # try common attrs
        if hasattr(m, "jack"):
            j = m.jack  # type: ignore[attr-defined]
            r = getattr(j.rank, "name", str(j.rank))
            s = getattr(j.suit, "name", str(j.suit))
            return f"{t}(jack={r}_{s})"
        return t

    wins = 0
    losses: List[Tuple[int, str]] = []

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("PerfectInfoBot vs RdeepBot — LOSS DIAGNOSTICS\n\n")

        for i in range(games):
            seed = seed0 + i

            # fresh bots each game (important so internal caches don't leak across games)
            pi_inner = PerfectInfoBot(name="PerfectInfo")
            rd_inner = RdeepBot(num_samples=rdeep_samples, depth=rdeep_depth, rand=random.Random(99991 + seed))

            pi = LoggingBot(pi_inner, "PI")
            rd = LoggingBot(rd_inner, "RD")

            winner, game_points, score = engine.play_game(pi, rd, random.Random(seed))

            pi_won = (winner == pi)
            if pi_won:
                wins += 1
            else:
                # record a loss report
                header = f"\n=== LOSS seed={seed} winner={winner} points={game_points} score={score} ===\n"
                f.write(header)
                f.write("\n-- Move log --\n")
                for line in pi.log:
                    f.write(line + "\n")
                for line in rd.log:
                    f.write(line + "\n")

                # also print it to console briefly
                losses.append((seed, header.strip()))
                print(f"LOSS at seed {seed}: points={game_points}, score={score}")

            # progress
            if (i + 1) % max(1, games // 10) == 0:
                print(f"Progress {i+1}/{games} | wins={wins} | winrate={wins/(i+1):.1%}")

        f.write(f"\nSUMMARY: wins={wins}/{games} ({wins/games:.1%})\n")
        if losses:
            f.write("\nLOSS SEEDS:\n")
            for seed, hdr in losses:
                f.write(f"- {seed}\n")

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
