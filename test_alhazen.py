# Copyright (c) 2020-2022 Carnegie Mellon University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from collections import defaultdict
from itertools import count
import math
from multiprocessing import current_process
from pytest import raises
import random
import statistics
import time

from alhazen import *


class Trivial(Experiment):

    def prepare_experiment(self, **kwargs):
        self.info = list()

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
        fuck = False
        with self.log as logwriter:
            fuck = True
            logwriter.write(f"{participant}{condition}{context['c']}{context['p']}")
        assert fuck
        return ((participant, condition, context["c"], context["p"], self.setup_called),
                current_process().name)

    def finish_participant(self, participant, condition, result):
        self.info.append((participant, condition))
        return result

    def finish_condition(self, condition, result):
        return result

    def finish_experiment(self, result):
        self.info.append("finished")
        return result


def test_trivial():
    tr = Trivial(show_progress=False)
    assert tr.participants == 1
    assert tr.conditions == (None,)
    assert tr.process_count > 0
    assert not tr.show_progress
    r = list(tr.run(a=True, b=17))
    assert len(r) == 1
    assert r[0][0] == (0, None, None, 0, 'setup called')
    with raises(RuntimeError):
        tr.run()
    tr = Trivial(show_progress=False,
                 conditions=(2**i for i in range(4)),
                 process_count=2,
                 participants=3)
    assert tr.participants == 3
    assert tr.conditions == (1, 2, 4, 8)
    assert tr.process_count == 2
    assert not tr.show_progress
    results = tr.run(a=False, c=-7)
    result_set = set()
    process_set = set()
    for c in tr.conditions:
        r = list(results[c])
        assert len(r) == 3
        for t, p in r:
            result_set.add(t)
            process_set.add(p)
    assert result_set == {(0, 1, 3, 0, "setup called"),
                          (1, 2, 6, 7, "setup called"),
                          (2, 8, 24, 14, "setup called"),
                          (0, 2, 6, 0, "setup called"),
                          (2, 4, 12, 14, "setup called"),
                          (1, 1, 3, 7, "setup called"),
                          (1, 4, 12, 7, "setup called"),
                          (1, 8, 24, 7, "setup called"),
                          (2, 1, 3, 14, "setup called"),
                          (0, 8, 24, 0, "setup called"),
                          (0, 4, 12, 0, "setup called"),
                          (2, 2, 6, 14, "setup called")}
    assert len(process_set) == 2

def test_log(tmp_path):
    p = tmp_path / "log.txt"
    tr = Trivial(show_progress=False,
                 conditions="abc",
                 participants=3,
                 logfile=p).run()
    with open(p) as f:
        assert len(f.readline()) == 57


class TrivialRandom(IteratedExperiment):

    def run_participant_prepare(self, participant, condition, context):
        # Sometimes delay to mix things up a bit
        if random.random() < 0.1:
           time.sleep(0.1 * random.random())

    def run_participant_run(self, round, participant, condition, context):
        return random.randint(0, 10**60)   # should be conflict-free :-)

    def run_participant_continue(self, round, participant, condition, context):
        return round < 50 or participant % 2 == 0

    def run_participant_finish(self, participant, condition, results):
        return results[0:90] if results and participant % 10 == 0 else results


def test_trivial_random():
    assert len(list(TrivialRandom(show_progress=False).run())) == 1
    r = list(TrivialRandom(participants=1000, rounds=100, show_progress=False).run())
    assert len(r) == 1000
    d = defaultdict(int)
    for x in r:
        assert len(x) in {50, 90, 100}
        d[len(x)] += 1
    assert d[50] == 500
    assert d[90] == 100
    assert d[100] == 400
    for n in [0, 1, 2, 3, 4, 10, 100]:
        r = list(TrivialRandom(participants=20, rounds=10, process_count=n, show_progress=False).run())
        assert len(r) == 20
        for x in r:
            assert len(x) == 10


class LogisticSampler(IteratedExperiment):

    def prepare_condition(self, condition, context):
        context["width"] = condition if condition is not None else 1

    def setup(self):
        # Yes, this is thoroughly silly, it's just herre to make sure setup()
        # gets called properly.
        self.random_source = random.random

    def run_participant_prepare(self, participant, condition, context):
        self.sum = 0

    def run_participant_run(self, round, participant, condition, context):
        rand = self.random_source()
        self.sum += context["width"] * math.log((1 - rand) / rand)
        return self.sum


def test_logistic_sampler():
    def execute_once(**kwargs):
        ls = LogisticSampler(participants=1000, rounds=100, show_progress=False, **kwargs)
        results = ls.run()
        for c in kwargs.get("conditions"):
            width = c if c is not None else 1
            res = list(results[c])
            stats = list()
            for i in range(len(res[0])):
                mean = statistics.mean(r[i] for r in res)
                stdev = statistics.pstdev((r[i] for r in res), mean)
                stats.append((mean, stdev))
            for t in stats:
                assert width * -2 < t[0] < width * 2
            assert width * (1.9 - 2) < stats[0][1] < width * (1.9 + 2)
            assert width * (17.6 - 2) < stats[99][1] < width * (17.6 + 2)
        with raises(KeyError):
            results[3]
    execute_once(process_count=1, conditions=[2])
    execute_once(process_count=2, conditions=[1000])
    execute_once(process_count=300, conditions=[0.0001])
    execute_once(conditions=[0.01, 1.1, 100.0])
