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

"""Alhazen is a small, simple framework to facilitate running experiments, written in
Python, using cognitive models, or similar applications, in multiple, parallel processes.
It is only useful on multi-core machines, though most modern machines are such; the more
cores, the more performance benefit you are likely to get by using it. It also depends
upon the experiment being structured as a large number of identical, independent runs of
the same activity, or of similar activities. This is a common pattern, each such run
usually corresponding to a distinct virtual participant, or possibly a collection of
interacting participants.

When an Alhazen experiment is run the initial process is used as a parent, controller
process and it spawns one or more child, worker processes to run the individual tasks.
Alhazen handles partitioning the sub-tasks between these workers, and collecting their
results, which can then be aggregated in the parent, controller process. To use Alhazen
you make a subclass of its :class:`Experiment` or :class:`IteratedExperiment` class,
override some of its methods to describe what the workers should do and what results to
return and aggregate, declare how many workers to use, and then call its :meth:`run`
method.

"""

__version__ = "1.2.1"

import os
import queue
import sys
import traceback
from multiprocessing import Process, Queue
from typing import Any, List

from tqdm import tqdm

TIMEOUT = 0.08


class Experiment:
    """An abstract base class, concrete subclasses of which define experiments that can be
    run a collection of independent tasks, possibly distributed to multiple worker
    processes when run on on a multi-core machine. A subclass of :class:`Experiment` must
    at least override the :meth:`run_participant` method; typically it will override one
    or more other methods.

    The *participants*, if supplied, it should be a positive integer, the number of
    virtual participants to run. If not supplied it defaults to 1.

    The *conditions*, if supplied, should be an iterable of values that are both hashable
    and `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
    These denote different conditions in which the task of the :class:`Experiment` should
    be run, and all *participants* are run once against each condition. Multiple,
    orthogonal sets of conditions are often most easily represented as tuples of elements
    of the underlying individual sets of conditions.

    The *process_count*, if supplied, should be a non-negative integer. If non-zero it
    is the number of worker processes to use. Note that the overall program will actually
    contain one more process than this, the control process, which is also the main
    process in which the :class:`Experiment`'s :meth:`run` method is called. If
    *process_count* is zero (the default if not supplied) it indicates that the number
    of worker processes to be used should be the number of cores available, as determined
    by Python's ``os`` module. Note that on Intel processors supporting them this count
    will include the "virtual" cores supplied by
    `Hyper-threading <https://en.wikipedia.org/wiki/Hyper-threading>`_. In this case it
    may sometimes be advantageous to use a lower number.

    By default when an :class:`Experiment` is running a
    `tqdm <https://tqdm.github.io/>`_ progress indicator is shown, advanced as each task
    is completed. This may be suppressed by setting *show_progress* to ``False``.

    """

    def __init__(self, participants=1, conditions=None, process_count=0, show_progress=True):
        self._participants = participants
        # The following disjunction is in case conditions is an iterator returning no objects;
        # such an iterator is truthy, but results in an empty tuple.
        self._conditions = (tuple(conditions) or (None,)) if conditions else (None,)
        self._process_count = min((process_count or len(os.sched_getaffinity(0))),
                                  (participants * len(self._conditions)))
        self._show_progress = show_progress
        self._results = {c: [None] * participants for c in self._conditions}

    @property
    def participants(self):
        """ The number of particpants specified when this :class:`Experiment` was created.
        This is a read only attribute and cannot be modified after the :class:`Experiment`
        is created.
        """
        return self._participants

    @property
    def conditions(self):
        """A tuple containing the conditions specified when this :class:`Experiment` was
        created. This is a read only attribute and cannot be modified after the
        :class:`Experiment` is created.
        """
        return self._conditions

    @property
    def process_count(self):
        """ The number of worker processes this :class:`Experiment` will use. This may
        differ from the number specified when the :class:`Experiment` was created, either
        because that number was zero, or because there are fewer actual tasks to perform.
        This is a read only attribute and cannot be modified after the :class:`Experiment`
        is created.
        """
        return self._process_count

    @property
    def show_progress(self):
        """Whether or not to show a progress indicator while this :class:`Experiment` is
        running. This is a read only attribute and cannot be modified after the
        :class:`Experiment` is created.
        """
        return self._show_progress

    def prepare_experiment(self, **kwargs):
       """The control process calls this method, once, before any of the other methods in
       the public API. If any keyword arguments were passed to to :class:`Experiment`'s
       :meth:`run` method, they are passed to this method. It can be used to allocate data
       structures or initialize other state required by the experiment. It can count on
       the :class:`Experiment`'s *process_count* slot have been initialized to the
       number of workers that will actually be used, as well as its *conditions* slot
       containing a list. This method is intended to be overridden in subclasses, and
       should not be called directly by the programmer. The default implementation of this
       method does nothing.
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
        tasks in the given *condition*. The *context* is a dictionary into which the
        method may write information that it wishes to pass to the task in the worker
        processes. Information added to the *context* must be
        `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
        This method is intended to be overridden in subclasses, and should not be called
        directly by the programmer. The default implementation of this method does
        nothing.
        """
        pass

    def prepare_participant(self, participant, condition, context):
        """The control process calls this method before asking a worker to execute a task
        on behalf of a *participant*, in this *condition*. The *participant* is an
        integer; participants are number sequentially, starting from zero. The *context*
        is a dictionary into which the method may write information that it wishes to pass
        to the task in the worker processes. Information added to the *context* must be
        `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
        The *context* contains any information added to it by
        :meth:`prepare_condition`, which is called before any calls to this method for a
        particular *condition*. The *context* passed to this method is a fresh copy of
        that potentially modified by :meth:`prepare_condition`, and does not contain any
        modifications made by earlier calls to :meth:`prepare_participant`. This method is
        intended to be overridden in subclasses, and should not be called directly by the
        programmer. The default implementation of this method does nothing.

        """
        pass

    def run_participant(self, participant, condition, context):
        """This is the principal method called in a worker process, and each call of this
        method executes the task of one participant in the given *condition*. The
        *participant* is a non-negative integer identifying the participant. The *context*
        is a dictionary possibly containing additional parameters or other information
        used by the tasks and provided by the :meth:`prepare_condition` and/or
        :meth:`prepare_participant` methods. This method typically returns a value, which
        is provided to the control process's :meth:`finish_participant` method. Any value
        :meth:`run_participant` returns must be
        `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
        This method must be overridden by subclasses, and should not be called directly by
        the programmer. The default implementation of this method raises a
        :Exc:`NotImplementedError`.
        """
        raise NotImplementedError("The run_participant() method must be overridden")

    def finish_participant(self, participant, condition, result):
        """The control process calls this method after each participant's task has been
        completed by a worker process. Passed as *results* is the value returned by the
        :meth:`run_participant` method in the worker process, or ``None`` if no value was
        returned. The value returned by this method, of None if none is returned, is
        stored for retrieval by the :meth:`results` method. The :meth:`finish_participant`
        method is intended to be overridden in subclasses, and should not be called
        directly by the programmer. The default implementation of this method returns
        the value of its *result* parameter unchanged.

        """
        return result

    def finish_experiment(self):
        """The control process calls this method, once, after all the participants have
        been run in each of the conditions, and the corresponding calls to
        :meth:`finish_participant` have all completed. This method is intended to be
        overridden in subclasses, and should not be called directly by the programmer. The
        default implementation of this method does nothing.
        """
        pass

    def results(self, condition=None):
        """The method is called by the programmer, and runs in the control process. It
        returns an iterator, ordered by participant number, of the results returned by
        :meth:`finish_participant`. If that method has not be overridden those will be the
        values returned by :meth:`run_participant`. If called before the experiment ends
        the iterator may return zero results, or fewer results than the number of
        participants, only those that have so far been processed. Raises a :exc:`KeyError`
        if *condition* is not a condition in this :class:`Experiment`.
        """
        return (r for r in self._results[condition])

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
        total_tasks = self._participants * len(self._conditions)
        self.prepare_experiment(**kwargs)
        task_q = Queue()
        result_q = Queue()
        processes = [ Process(target=self._run_one, args=[task_q, result_q])
                      for i in range(self._process_count) ]
        for p in processes:
            p.start()
        try:
            tasks = ((c, p) for c in self._conditions for p in range(self._participants))
            tasks_completed = 0
            self._progress = self._show_progress and tqdm(total=total_tasks)
            condition_context = None
            current_condition = None
            participant = None
            results = { c: [None] * self._participants for c in self._conditions }
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
                        self._results[c][p] = self.finish_participant(p, c, result)
                        tasks_completed += 1
                        did_something = True
                        if self._progress:
                            self._progress.update()
                    except queue.Empty:
                        break
                blocking = not did_something
            for i in range(len(processes)):
                task_q.put((None, None, None))
            self.finish_experiment()
            for p in processes:
                p.join()
        except:
            traceback.print_exc()
            for p in processes:
                try:
                    p.terminate
                except:
                    traceback.print_exc()
        finally:
            result_q.close()
            task_q.close()
            if self._progress:
                self._progress.close()
        return self

    def _run_one(self, task_q, result_q):
        # called in the child processes
        try:
            self.setup()
            while True:
                participant, condition, context = task_q.get()
                if participant is None:
                    break
                result = self.run_participant(participant, condition, context)
                result_q.put((participant, condition, result))
        except:
            traceback.print_exc()
            sys.exit(1)


class IteratedExperiment(Experiment):
    """This is a an abstract base class, a subclass of :class:`Experiment`, for
    experiements where each participant performaces sequence of identical or similar
    actions, one per round. The *rounds* is the maximum number of rounds that will be
    executted. If :meth:`run_participant_continue` is overriden is is possible that fewer
    than *rounds* rounds will be executed. The *rounds* should be a positive integer, and,
    if not supplied, defaults to 1.

    This subclass overrides :meth:`run_participant`. Typically the programmer will not
    override :meth:`run_participant` themself, but if they do, they should generally
    be sure to call the superclass's (that is, :class:`IteratedExperiment`'s) version,
    and return the value it returns. :class:`IteratedExperiment`'s :meth:`run_participant`
    decomposes this activity into four finer grained methods, all called in the worker
    process: :meth:`run_participant_prepare`, :meth:`run_participant_run`,
    :meth:`run_participant_continue`, and :meth:`run_participant_finish`, all inteded for
    overriding. The programmer must override at least :meth:`run_participant_run`, which
    is called repeated, once for each round, and should return a
    `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_ value
    which which is accumlated into a list, indexed by round. This list is returned to the
    parent, control process as the result for the participant and condition.

    As a subclass of :class:`Experiment` the other methods and attributes of that parent
    class are, of course, also available.
    """

    def __init__(self, rounds=1, **kwargs):
        super().__init__(**kwargs)
        self._rounds = rounds

    @property
    def rounds(self):
        """ The maximum number of rounds specified when this :class:`IteratedExperiment`
        was created. This is a read only attribute and cannot be modified after the
        :class:`IteratedExperiment` is created.
        """
        return self._rounds

    def run_participant_prepare(self, participant, condition, context):
        """This method is called at the start of a worker process running a participant's
        activity, before the loop in which :meth:`run_participant_run` is called. Its
        arguments are as for :meth:`run_participant`. Any changes it makes to *context*
        will be visible to future calls to :meth:`run_participant_continue`,
        :meth:`run_participant_run` and :meth:`run_participant_finish` by this participant
        in this condtion, but not any others. This method is intended to be overridden in
        subclasses, and should not be called directly by the programmer. The default
        implementation of this method does nothing.
        """
        pass

    def run_participant_continue(self, round, participant, condition, context):
        """This method is called in a worker process before each call of
        :meth:`run_participant_run`. If it returns ``True`` (or any truthy value) the
        iterations continue and :meth:`run_participant_run` is called. If it returns
        ``False`` (or any falsey value) this participants activities in this condition end
        with no more rounds. The values of *round*, *participant*, *condition* and
        *context* are as for :meth:`run_participant_run`. Any changes made to the
        *context* by :meth:`run_participant_prepare` or by previous invocations of
        :meth:`run_participant_run` or :meth:`run_participant_continue` are retained in
        the *context* presented to this method, and any changes this method makes to its
        *context* are propogated to future calls of :meth:`run_participant_continue`,
        :meth:`run_participant_run` and :meth:`run_participant_finish` by this participant
        in this condition, but not to any others. This method is intended to be overridden
        in subclasses, and should not be called directly by the programmer. The default
        implementation of this method returns ``True``.
        """
        return True

    def run_participant_run(self, round, participant, condition, context):
        """This method should be overriden to perform one round's worth of activity by the
        participant, in a worker process. The *round* is a non-negative integer which
        round this is; it starts at zero and increases by one at each iteration. The
        *participant*, *condition* and *context* are as for :meth:`run_participant`. Any
        changes made to the *context* by :meth:`run_participant_prepare` or by previous
        invocations of :meth:`run_participant_run` or :meth:`run_participant_continue` are
        retained in the *context* presented to this method, and any changes this method
        makes to its *context* are propogated to future calls of
        :meth:`run_participant_continue`, :meth:`run_participant_run` and
        :meth:`run_participant_finish` by this participant in this condition, but not to
        any others. This method should return a
        `picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_
        value which will be collected into a list which list will be the value returned to
        the control process for this participant and condition. This method is intended to
        be overridden in subclasses, and should not be called directly by the programmer.
        The default implementation of this method raises a :exc:`NotImplementedError`.
        """
        raise NotImplementedError("The run_participant_run() method must be overridden")

    def run_participant_finish(self, participant, condition, context):
        """This method is called after all the rounds for a participant in a condition
        have been executed. The *participant*, *condition* and *context* are as for
        :meth:`run_participant`. Any changes made to the *context* by
        :meth:`run_participant_prepare` or by previous invocations of
        :meth:`run_participant_run` or :meth:`run_participant_continue` are retained in
        the *context* presented to this method This method is intended to be overridden in
        subclasses, and should not be called directly by the programmer. The default
        implementation of this method does nothing.
        """
        pass

    def run_participant(self, participant, condition, context):
        results = []
        self.run_participant_prepare(participant, condition, context)
        for round in range(self.rounds):
            if not self.run_participant_continue(round, participant, condition, context):
                break
            results.append(self.run_participant_run(round, participant, condition, context))
        self.run_participant_finish(participant, condition, context)
        return tuple(results)
