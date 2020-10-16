# Copyright (c) 2020 Carnegie Mellon University
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

"""Alhazen is a simple framework to facilitate running experiments, written in
Python, using cognitive models, or similar applications, in multiple, parallel
processes. It is only useful on multi-core machines, though most modern machines
are such; the more cores, the more performance benefit you are likely to get by
using it. It also depends upon the experiment being structured as a large number
of identical, independent runs of the same activity, or of similar activities.
This is a common pattern, each such run usually corresponding to a distinct
virtual participant, or possibly a collection of interacting participants.

When an Alhazen experiment is run the initial process is used as a parent,
controller process and it spawns one or more child, worker processes to run the
individual tasks. Alhazen handles partitioning the sub-tasks between these
workers, and collecting their results, which can then be aggregated in the
parent, controller process. To use Alhazen you make a subclass of its
:class:`Experiment` class, override some of its methods to describe what the
workers should do and how to aggregate their results, declare how many workers
to use, and then call its :meth:`run` method.
"""

__version__ = "1.0"

import os
import queue
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Any, List

from tqdm import tqdm

TIMEOUT = 0.08


@dataclass
class Experiment:
    """An abstract base class, concrete subclasses of which define experiments that can be
    run a collection of independent tasks, possibly distributed to multiple worker
    processes when run on on a multi-core machine. A subclass of :class:`Experiment` must
    at least override the :meth:`run_participant` method; typically it will override one
    or more other methods.

    The ``participants``, if supplied, it should be a positive integer, the number of
    virtual participants to run. If not supplied it defaults to 1.

    The ``conditions``, if supplied, should be an iterable of
    `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
    values. These denote different conditions in which the task of the :class:`Experiment`
    should be run, and all ``participants`` are run once against each condition.
    Besides being picklable, conditions are usually hashable as well, to allow easy
    accumulation of data in dictionaries, though this is not required. Multiple,
    orthogonal conditions are often most easily represented as tuples of the underlying
    individual sets of conditions.

    The ``rounds``, if supplied, should be a positive integer. It is supplied as a
    suggestion to the tasks in the worker processes, denoting how many iterations to run
    of an iterable task. Its use is not required, but is often convenient. If not supplied
    its default value is 1.

    The ``process_count``, if supplied, should be a non-negative integer. If non-zero it
    is the number of worker processes to use. Note that the overall program will actually
    contain one more process than this, the control process, which is also the main
    process in which the :class:`Experiment`'s :meth:`run` method is called. If
    ``process_count`` is zero it indicates that the number of work processes to be used
    should be the number of cores available, as determined by Python's ``os`` module.
    Note that on Intel processors supporting them this count will include the "virtual"
    cores supplied by `Hyper-threading <https://en.wikipedia.org/wiki/Hyper-threading>`_.
    In this case it may sometimes be advantageous to use a lower number.

    By default when an :class:``Experiment`` is running a
    `tqdm <https://tqdm.github.io/>`_ progress indicator is shown, advanced as each task
    is completed. This may be suppressed by setting ``show_progress`` to ``False``.

    """

    participants: int = 1
    conditions: List[Any] = field(default_factory=list)
    rounds: int = 1
    process_count: int = 0
    show_progress: bool = True

    def prepare_experiment(self, **kwargs):
       """The control process calls this method, once, before any of the other methods in
       the public API. If any keyword arguments were passed to to
       :class:`Experiment`'s :meth:`run` method, they are passed to this
       method. It can be used to allocate data structures or initialize other
       state required by the experiment. It can count on the
       :class:`Experiment`'s ``process_count`` slot have been initialized to
       the number of workers that will actually be used, as well as its
       ``conditions`` slot containing a list. This method is intended to be
       overridden in subclasses, and should not be called directly by the
       programmer. The default implementation of this method does nothing.
       """
       pass

    def setup(self):
        """Each worker process calls this method, once, before performing the
        work of any participants for any condtion. It can be used to allocate data
        structures or initialize other state required by the worker processes. This method
        is intended to be overridden in subclasses, and should not be called directly by
        the programmer. The default implementation of this method does nothing.
        """
        pass

    def prepare_condition(self, condition, context):
        """The control process calls this method before asking the workers to execute
        tasks in the given ``condition``. The ``context`` is a dictionary into which the
        method may write information that it wishes to pass to the task in the worker
        processes. Information added to the ``context`` must be
        `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
        This method is intended to be overridden in subclasses, and should not be called
        directly by the programmer. The default implementation of this method does
        nothing.
        """
        pass

    def prepare_participant(self, participant, condition, context):
        """The control process calls this method before asking a worker to execute a task
        on behalf of a ``participant``, in this ``condition``. The ``participant`` is an
        integer; participants are number sequentially, starting from zero. The ``context``
        is a dictionary into which the method may write information that it wishes to pass
        to the task in the worker processes. Information added to the ``context`` must be
        `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
        The ``context`` contains any information added to it by
        :meth:`prepare_condition`, which is called before any calls to this method for a
        particular ``condition``. The ``context`` passed to this method is a fresh copy of
        that potentially modified by :meth:`prepare_condition`, and does not contain any
        modifications made by earlier calls to :meth:`prepare_participant`. This method is
        intended to be overridden in subclasses, and should not be called directly by the
        programmer. The default implementation of this method does nothing.

        """
        pass

    def run_participant(self, rounds, participant, condition, context):
        """This is the principal method called in a worker process, and each call of this
        method executes the task of one participant in the given ``condition``. The
        ``participant`` is a non-negative integer identifying the participant. The value
        of ``rounds``, a positive integer, is a suggestion of how many rounds to run an
        iterative task, as supplied when the :class:`Experiment` was defined; however, the
        task need not be iterative, and might define a different way to terminate than as
        simple counter of rounds, in which case ``rounds`` may be ignored. The ``context``
        is a dictionary possibly containing additional parameters or other information
        used by the tasks and provided by the :meth:`prepare_condition` and/or
        :meth:`prepare_participant` methods. This method typically returns a value, which
        is provided to the control process's :meth:`finish_participant` method. Any value
        :meth:`run_participant` returns must be
        `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
        This method must be overridden by subclasses, and should not be called directly
        by the programmer. The default implementation of this method raises a
        :Exc:`NotImplementedError`.

        """
        raise NotImplementedError("The run_participant() method must be overridden")

    def finish_participant(self, participant, condition, result):
        """The control process calls this method after each participant's task has been
        completed by a worker process. Passed as ``results`` is the value returned by the
        :meth:`run_participant` method in the worker process, or ``None`` if no value was
        returned. This method is often used for aggregating results from the workers'
        actions. This method is intended to be overridden in subclasses, and should not be
        called directly by the programmer. The default implementation of this method does
        nothing.
        """
        pass

    def finish_experiment(self):
        """The control process calls this method, once, after all the participants have
        been run in each of the conditions, and the corresponding calls to
        :meth:`finish_participant` have all completed. This method is intended to be
        overridden in subclasses, and should not be called directly by the programmer. The
        default implementation of this method does nothing.
        """
        pass

    def run(self, **kwargs):
        """This method is called by the programmer to begin processing of the various
        tasks of this :class:`Experiment`. It creates one or more worker processes, and
        partitions tasks between them ensuring that one, and exactly one, worker process
        executes a task only once for each pairing of a participant and a condition, for
        all the participants and all the conditions. The :meth:`run_participant` method
        must have been overridden to define the task to be run in the worker processes.
        Typically other methods are overridden to aggregate the results of these tasks,
        and possibly to setup data structures and other state required by them. If any
        keyword arguments are supplied when calling :meth:`run` they are passed to the
        :class:`Experiment`'s :meth:`prepare_experiment` method. Returns this
        :class:`Experiment`.
        """
        self.conditions = self.conditions or (None,)
        total_tasks = self.participants * len(self.conditions)
        self.process_count = min((self.process_count or len(os.sched_getaffinity(0))),
                                 total_tasks)
        self.prepare_experiment(**kwargs)
        task_q = Queue()
        result_q = Queue()
        processes = [ Process(target=self.run_one, args=[task_q, result_q])
                      for i in range(self.process_count) ]
        try:
            for p in processes:
                p.start()
            tasks = ((c, p) for c in self.conditions for p in range(self.participants))
            tasks_completed = 0
            self.progress = self.show_progress and tqdm(total=total_tasks)
            condition_context = None
            current_condition = None
            participant = None
            results = { c: [None] * self.participants for c in self.conditions }
            blocking = False
            while tasks_completed < total_tasks:
                did_something = False
                if participant is None:
                    try:
                        condition, participant = next(tasks)
                    except StopIteration:
                        participant = None
                if participant is not None:
                    if not condition_context or condition != current_condition:
                        condition_context = dict()
                        current_condition = condition
                        self.prepare_condition(condition, condition_context)
                    participant_context = dict(condition_context)
                    self.prepare_participant(participant, condition, participant_context)
                    try:
                        task_q.put((participant, condition, participant_context), blocking, TIMEOUT)
                        participant = None
                    except queue.Full:
                        pass
                    did_something = True
                while True:
                    try:
                        p, c, result = result_q.get(blocking, TIMEOUT)
                        self.finish_participant(p, c, result)
                        tasks_completed += 1
                        did_something = True
                        if self.progress:
                            self.progress.update()
                    except queue.Empty:
                        break
                blocking = not did_something
            for i in range(len(processes)):
                task_q.put((None, None, None))
            self.finish_experiment()
        finally:
            for p in processes:
                p.join()
            result_q.close()
            task_q.close()
            if self.progress:
                self.progress.close()
        return self

    def run_one(self, task_q, result_q):
        # called in the child processes
        self.setup()
        while True:
            participant, condition, context = task_q.get()
            if participant is None:
                break
            result = self.run_participant(self.rounds, participant, condition, context)
            result_q.put((participant, condition, result))
