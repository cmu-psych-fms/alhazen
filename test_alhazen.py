from itertools import count
import math
import random
import statistics
import time

from alhazen import Experiment


class TrivialRandom(Experiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = set()

    def run_participant(self, rounds, participant, condition, context):
        # Sometimes delay to mix things up a bit
        if random.random() < 0.1:
           time.sleep(0.1 * random.random())
        results = []
        for i in range(rounds):
            results.append(random.randint(0, 10**60)) # should be conflict-free :-)
        return results

    def finish_participant(self, participant, condition, result):
        for n in result:
            self.results.add(n)


def test_trivial_random():
    tr = TrivialRandom(show_progress=False)
    tr.run()
    assert len(tr.results) == 1
    tr = TrivialRandom(participants=1000, rounds=100, show_progress=False)
    tr.run()
    assert len(tr.results) == 1000 * 100
    for n in [0, 1, 2, 3, 4, 10, 100]:
        tr = TrivialRandom(participants=20, rounds=10, process_count=n, show_progress=False)
        tr.run()
        assert len(tr.results) == 20 * 10


class LogisticSampler(Experiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = dict()

    def prepare_condition(self, condition, context):
        # No, this doesn't need to be done in prepare_conditon, it could easily be
        # done elsewhere, but we're doing it here to make sure prepare_condition()
        # is called properly.
        context["width"] = condition if condition is not None else 1
        self.results[condition] = [list() for i in range(self.rounds)]

    def setup(self):
        # Yes, this is thoroughly silly, it's just herre to make sure setup()
        # gets called properly.
        self.random_source = random.random

    def run_participant(self, rounds, participant, condition, context):
        results = []
        sum = 0
        for r in range(rounds):
            rand = self.random_source()
            sum += context["width"] * math.log((1 - rand) / rand)
            results.append(sum)
        return results

    def finish_participant(self, participant, condition, result):
        for r, i in zip(result, count()):
            self.results[condition][i].append(r)

    def finish_experiment(self):
        for c in self.results:
            for r, i in zip(self.results[c], count()):
                mean = statistics.mean(r)
                stdev = statistics.pstdev(r, mean)
                self.results[c][i] = (mean, stdev)



def test_logistic_sampler():
    def execute_once(**kwargs):
        ls = LogisticSampler(participants=1000, rounds=100, show_progress=False, **kwargs)
        ls.run()
        for c in kwargs.get("conditions", [None]):
            width = c if c is not None else 1
            for r in ls.results[c]:
                assert width * -2 < r[0] < width * 2
            assert width * (1.9 - 2) < ls.results[c][0][1] < width * (1.9 + 2)
            assert width * (17.6 - 2) < ls.results[c][99][1] < width * (17.6 + 2)
    execute_once()
    execute_once(process_count=1, conditions=[2])
    execute_once(process_count=2, conditions=[1000])
    execute_once(process_count=300, conditions=[0.0001])
    execute_once(conditions=[0.01, 1.1, 100.0])
