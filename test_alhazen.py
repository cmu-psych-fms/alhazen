from itertools import count
import math
from multiprocessing import current_process
import random
import statistics
import time

from alhazen import Experiment


class Trivial(Experiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = set()
        self.process_names = set()

    def prepare_experiment(self, **kwargs):
        self.results.add(tuple(kwargs.items()))

    def setup(self):
        self.setup_called = "setup called"

    def prepare_condition(self, condition, context):
        try:
            condition *= 3
        except TypeError:
            pass
        context["c"] = condition

    def prepare_participant(self, participant, condition, context):
        context["p"] = participant * 7

    def run_participant(self, participant, condition, context):
        return ((participant, condition, context["c"], context["p"], self.setup_called),
                current_process().name)

    def finish_participant(self, participant, condition, result):
        self.results.add(result[0])
        self.process_names.add(result[1])

    def finish_experiment(self):
        self.results.add("experiment finished")


def test_trivial():
    tr = Trivial(show_progress=False)
    assert tr.participants == 1
    assert tr.conditions == (None,)
    assert tr.rounds == 1
    assert tr.process_count > 0
    assert not tr.show_progress
    assert tr.run(a=True, b=17) is tr
    assert tr.results == {(('a', True), ('b', 17)),
                          (0, None, None, 0, 'setup called'),
                          'experiment finished'}
    assert len(tr.process_names) == 1
    tr = Trivial(show_progress=False,
                 conditions=(2**i for i in range(4)),
                 process_count=2,
                 participants=3,
                 rounds=4)
    assert tr.participants == 3
    assert tr.conditions == (1, 2, 4, 8)
    assert tr.rounds == 4
    assert tr.process_count == 2
    assert not tr.show_progress
    assert tr.run(a=False, c=-7) is tr
    assert tr.results == {(0, 8, 24, 0, 'setup called'),
                          (0, 1, 3, 0, 'setup called'),
                          (2, 2, 6, 14, 'setup called'),
                          'experiment finished',
                          (0, 2, 6, 0, 'setup called'),
                          (2, 4, 12, 14, 'setup called'),
                          (2, 1, 3, 14, 'setup called'),
                          (2, 8, 24, 14, 'setup called'),
                          (0, 4, 12, 0, 'setup called'),
                          (('a', False), ('c', -7)),
                          (1, 1, 3, 7, 'setup called'),
                          (1, 8, 24, 7, 'setup called'),
                          (1, 2, 6, 7, 'setup called'),
                          (1, 4, 12, 7, 'setup called')}
    assert len(tr.process_names) == 2


class TrivialRandom(Experiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = set()

    def run_participant(self, participant, condition, context):
        # Sometimes delay to mix things up a bit
        if random.random() < 0.1:
           time.sleep(0.1 * random.random())
        results = []
        for i in range(self.rounds):
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

    def run_participant(self, participant, condition, context):
        results = []
        sum = 0
        for r in range(self.rounds):
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
