from alhazen import IteratedExperiment
import click
from itertools import count
import matplotlib.pyplot as plt
import pyactup
import random

EXPECTED_VALUES = [5, 4, 3, 2, 1]

class SafeRisky(IteratedExperiment):

    def run_participant_prepare(self, participant, condition, context):
        self.memory = pyactup.Memory()
        self.memory.learn({"choice": "safe", "payoff": 12})
        self.memory.learn({"choice": "risky", "payoff": 12}, advance=True)

    def run_participant_run(self, round, participant, condition, context):
        choice, bv = self.memory.best_blend("payoff", ("safe", "risky"), "choice")
        if choice == "safe":
            payoff = 1
        elif random.random() < condition / 10:
            payoff = 10
        else:
            payoff = 0
        self.memory.learn({"choice": choice, "payoff": payoff}, advance=True)
        self.log([condition, participant, round, choice, payoff])
        return choice == "risky"


# Note that if run with the default 10,000 participants and 200 rounds the log file
# will consist of nearly 10 million lines totalling nearly 200 megabytes.

@click.command()
@click.option("--rounds", default=200, help="the number of rounds each participant plays")
@click.option("--participants", default=10_000, help="the number of participants")
@click.option("--workers", default=0,
              help="number of worker processes, zero (the default) means as many as available cores")
@click.option("--log", help="a log file to which to write details of the experiment")
def main(rounds=200, participants=10_000, workers=0, log=None):
    exp = SafeRisky(rounds=rounds,
                    conditions=EXPECTED_VALUES,
                    participants=participants,
                    process_count=workers,
                    logfile=log,
                    csv=True,
                    fieldnames=("expected value,participant,round,choice,payoff".split(",")))
    results = exp.run()
    for c in exp.conditions:
        plt.plot(range(1, rounds + 1),
                 list(sum(r[i] for r in results[c]) / participants
                          for i in range(rounds)),
                 label=f"Risky EV = {c}")
    plt.legend()
    plt.xlabel("round")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("fraction choosing risky")
    plt.title(f"Safe versus Risky, {participants:,d} participants")
    plt.show()


if __name__== "__main__":
    main()
