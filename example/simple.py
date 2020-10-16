from alhazen import Experiment
import click
from itertools import count
import matplotlib.pyplot as plt
import pyactup
import random

ROUNDS = 200
PARTICIPANTS = 10_000

class SafeRisky(Experiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = [0] * self.rounds

    def run_participant(self, rounds, participant, condition, context):
        risky_chosen = [True] * rounds
        mem = pyactup.Memory()
        mem.learn(choice="safe", payoff=12, advance=0)
        mem.learn(choice="risky", payoff=12)
        for r in range(rounds):
            choice, blended_value = mem.best_blend("payoff", ("safe", "risky"), "choice")
            if choice == "safe":
                payoff = 1
                risky_chosen[r] = False
            elif random.random() < 0.1:
                payoff = 10
            else:
                payoff = 0
            mem.learn(choice=choice, payoff=payoff)
        return risky_chosen

    def finish_participant(self, participant, condition, result):
        for r, i in zip(result, count()):
            if r:
                self.results[i] += r / self.participants


@click.command()
@click.option("--workers", "-w", default=0,
              help="number of worker processes, zero (the default) means as many as available cores")
def main(**kwargs):
    sr = SafeRisky(participants=PARTICIPANTS, rounds=ROUNDS, process_count=kwargs["workers"])
    sr.run()
    plt.plot(range(1, sr.rounds + 1), sr.results)
    plt.xlabel("round")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("fraction choosing risky")
    plt.title(f"Safe versus Risky, {sr.participants:,d} participants")
    plt.show()


if __name__== "__main__":
    main()
