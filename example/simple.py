from alhazen import IteratedExperiment
import click
import matplotlib.pyplot as plt
import pyactup
import random


class SafeRisky(IteratedExperiment):

    def run_participant_prepare(self, participant, condition, context):
        self.memory = pyactup.Memory()
        self.memory.learn({"choice": "safe", "payoff": 12})
        self.memory.learn({"choice": "risky", "payoff": 12}, advance=True)

    def run_participant_run(self, round, participant, condition, context):
        choice, bv = self.memory.best_blend("payoff", ("safe", "risky"), "choice")
        if choice == "safe":
            payoff = 1
        elif random.random() < 0.1:
            payoff = 10
        else:
            payoff = 0
        self.memory.learn({"choice": choice, "payoff": payoff}, advance=True)
        return choice == "risky"


@click.command()
@click.option("--rounds", default=200, help="the number of rounds each participant plays")
@click.option("--participants", default=10_000, help="the number of participants")
@click.option("--workers", default=0,
              help="number of worker processes, zero (the default) means as many as available cores")
def main(rounds=200, participants=10_000, workers=0):
    exp = SafeRisky(rounds=rounds,
                    participants=participants,
                    process_count=workers)
    results = exp.run()
    plt.plot(range(1, rounds + 1),
             list(sum(r[i] for r in results) / participants
                      for i in range(rounds)))
    plt.xlabel("round")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("fraction choosing risky")
    plt.title(f"Safe versus Risky, {participants:,d} participants")
    plt.show()


if __name__== "__main__":
    main()
