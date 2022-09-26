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

__version__ = "1.3.4"

import csv
import os
import queue
import sys
import traceback
from collections import defaultdict
from contextlib import contextmanager
from multiprocessing import Lock, Process, Queue
from typing import Any, List

import psutil
from tqdm import tqdm

TIMEOUT = 0.08


def _available_processors():
    try:
        return psutil.cpu_count(logical=False)
    except Exception:
        return 4


class Experiment:
    """An abstract base class, concrete subclasses of which define experiments that can be
    run a collection of independent tasks, possibly distributed to multiple worker
    processes when run on on a multi-core machine. A subclass of :class:`Experiment` must
    at least override the :meth:`run_participant` method; typically it will override one
    or more other methods.

    The *participants*, if supplied, it should be a positive integer, the number of
    virtual participants to run. If not supplied it defaults to 1.

    The *conditions*, if supplied, it should be an iterable of values that are both
    hashable and `picklable
    <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_. These denote
    different conditions in which the task of the :class:`Experiment` should be run, and
    all *participants* are run once against each condition. Multiple, orthogonal sets of
    conditions are often most easily represented as tuples of elements of the underlying
    individual sets of conditions.

    The *process_count*, if supplied, should be a non-negative integer. If non-zero it is
    the number of worker processes to use. Note that the overall program will actually
    contain one more process than this, the control process, which is also the main
    process in which the :class:`Experiment`'s :meth:`run` method is called. If
    *process_count* is zero (the default if not supplied) it indicates that Alhazen should
    attempt to determine the number of cores available and use this number of worker
    processes. For various reasons this determination may be inaccurate, so if you know
    by other means what will work best for your experiment it may be better to supply it
    explicitly. Note also that this determination is of physical cores, ignoring any
    virtual ones supplied by `simultaneous mutlithreading
    <https://en.wikipedia.org/wiki/Simultaneous_multithreading>`_. If such a determination
    cannot be made a default of 4 is used.

    By default when an :class:`Experiment` is running a
    `tqdm <https://tqdm.github.io/>`_ progress indicator is shown, advanced as each task
    is completed. This may be suppressed by setting *show_progress* to ``False``.

    Because participants are typically run in multiple processes, if they all want to
    write to a single log file access to that file must be synchronized. Alhazen supplies
    a mechanism to more easily facilitate this in easy cases. A file name can be supplied
    as a value of the *logfile* parameter, in which case it names a file that will be
    opened for writing when the experiment is run, and closed when it finishes. Within the
    various methods the programmer overrides that file can be accessed using the
    :attr:`log` context manager, which ensures synchronization and makes available the
    file. If no *logfile* is supplied or it is None, the context manager returns a file
    that is a sink; that is, in this case writing to this file simply discards the output.

    It is frequently useful to write log files as Comma Separated Values (CSV) files. The
    *logfile* will be wrapped with a Python :class:`csv.writer` if *csv* is not false.
    If *csv* is a string it should be a CSV dialect. If it is ``True`` the `excel` dialect
    will be used. To use a :class:`csv.DictWriter` supply the *fieldnames*. In this case
    the *restval* and *extrasaction* parameters can be used to pass these extra parameters
    to the :class:`csv.DictWriter`.
    """

    def __init__(self,
                 participants=1,
                 conditions=None,
                 process_count=0,
                 show_progress=True,
                 logfile=None,
                 csv=None,
                 fieldnames=[],
                 restval="",
                 extrasaction="raise"):
        self._has_been_run = False
        self._participants = participants
        # The following disjunction is in case conditions is an iterator returning no objects;
        # such an iterator is truthy, but results in an empty tuple.
        self._conditions = (tuple(conditions) or (None,)) if conditions else (None,)
        self._process_count = min((process_count or _available_processors()),
                                  (participants * len(self._conditions)))
        self._show_progress = show_progress
        self._results = {c: [None] * participants for c in self._conditions}
        self._task_q = Queue()
        self._result_q = Queue()
        self._logfile_name = logfile or os.devnull
        self._logfile = None
        self._logfile_opened = False
        self._logwriter = None
        self._loglock = Lock()
        if isinstance(csv, str):
            self._csv = csv
        elif bool(csv) or fieldnames:
            self._csv = "excel"
        else:
            self._csv = None
        self._fieldnames = fieldnames
        self._restval = restval
        self._extrasaction = extrasaction

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
        completed by a worker process. Passed as *result* is the value returned by the
        :meth:`run_participant` method in the worker process, or ``None`` if no value was
        returned. The value returned by this method, or ``None`` if none is returned, is
        stored for passing to the :meth:`finish_condition` method when it is eventually
        called.. The :meth:`finish_participant` method is intended to be overridden in
        subclasses, and should not be called directly by the programmer. The default
        implementation of this method returns the value of its *result* parameter
        unchanged.
        """
        return result

    def finish_condition(self, condition, results):
        """The control process calls this method after all the participant's performing
        the task in a particular condition have finished and :meth:`finish_participant`
        has been called for them. This method is called only once for each condition.
        Passed as *results* is a list of results returned by the calls of the
        :meth:`finish_particpant` method. The value returned by this method, or ``None``
        if none is returned, is stored for passing to :meth:`finish_experiment` when the
        tasks for all conditions have been finished. The :meth:`finish_condition` method
        is intended to be overridden in subclasses, and should not be called directly by
        the programmer. The default implementation of this method returns the value of its
        *results* parameter unchanged.
        """
        return results

    def finish_experiment(self, results):
        """The control process calls this method, once, after all the participants have
        been run in each of the conditions, and the corresponding calls to
        :meth:`finish_condition` have all completed. This method is intended to be
        overridden in subclasses, and should not be called directly by the programmer.
        Passed as *results* is a dictionary indexed by the experiment's conditions, the
        values being the corresponding value returned by the call to
        :meth:`finish_condition`. The value returned by this method, or ``None`` if none
        is returned, is returned by the :class:`Experiment`'s :meth:`run` method. The
        :meth:`finish_exerpiment` method is intended to be overridden in subclasses, and
        should not be called directly by the programmer. The default implementation of
        this method returns *results* unchanged.
        """
        return results

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
        :class:`Experiment`'s :meth:`prepare_experiment` method. Returns the value
        returned by the :meth:`finish_experiment` method, or ``None``.
        """
        if self._has_been_run:
            raise RuntimeError(f"This Experiment has already been run")
        self._has_been_run = True
        total_tasks = self._participants * len(self._conditions)
        try:
            with open(self._logfile_name, "w"):
                # zaps any existing file
                pass
            self._open_log()
            self.prepare_experiment(**kwargs)
            self._close_log()
            processes = [ Process(target=self._run_one) for i in range(self._process_count) ]
            for p in processes:
                p.start()
            self._open_log()
            try:
                tasks = ((c, p) for c in self._conditions for p in range(self._participants))
                tasks_completed = 0
                condition_completions = defaultdict(int)
                self._progress = self._show_progress and tqdm(total=total_tasks)
                condition_context = None
                current_condition = None
                participant = None
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
                            self._task_q.put((participant, condition, participant_context), blocking, TIMEOUT)
                            participant = None
                        except queue.Full:
                            pass
                        did_something = True
                    while True:
                        try:
                            p, c, result = self._result_q.get(blocking, TIMEOUT)
                            self._results[c][p] = self.finish_participant(p, c, result)
                            tasks_completed += 1
                            condition_completions[c] += 1
                            assert condition_completions[c] <= self._participants
                            if condition_completions[c] == self._participants:
                                self._results[c] = self.finish_condition(c, self._results[c])
                            did_something = True
                            if self._progress:
                                self._progress.update()
                        except queue.Empty:
                            break
                    blocking = not did_something
                for i in range(len(processes)):
                    self._task_q.put((None, None, None))
                self._results = self.finish_experiment(self._results)
                for p in processes:
                    p.join()
            except:
                traceback.print_exc()
                for p in processes:
                    try:
                        p.terminate()
                    except:
                        traceback.print_exc()
            finally:
                self._result_q.close()
                self._task_q.close()
                if self._progress:
                    self._progress.close()
            return self._results if self._conditions != (None,) else self._results[None]
        finally:
            self._close_log()

    def _open_log(self):
        self._logfile = open(self._logfile_name, "a",
                             newline=("" if self._csv or self._fieldnames else None))
        self._logfile_opened = True
        if self._fieldnames:
            self._logwriter = csv.Dictwriter(self._logfile,
                                             self._fieldnames,
                                             restval=self._restval,
                                             extrasaction=self._extrasaction,
                                             dialect=self._csv)
        elif self._csv:
            self._logwriter = csv.writer(self._logfile, dialect=self._csv)

    def _close_log(self):
        if self._logfile_opened:
            self._logfile.close()
            self._logfile_opened = False
        self._logfile = None

    def _run_one(self):
        # called in the child processes
        try:
            self._open_log()
            self.setup()
            while True:
                participant, condition, context = self._task_q.get()
                if participant is None:
                    break
                result = self.run_participant(participant, condition, context)
                self._result_q.put((participant, condition, result))
        except:
            traceback.print_exc()
            sys.exit(1)
        finally:
            self._close_log()

    @property
    @contextmanager
    def log(self):
        """A context manager holding a lock to synchronize writing to this
        :class:`Experiment`'s log file. When this context manager is used in a
        ``with...as`` statement it in turn returns either the log file, or, if a CSV
        writer is being used, that writer. For example,

        with self.log as writer:
            writer.writerow([field, another_field])

        """
        with self._loglock:
            yield self._logwriter or self._logfile
            self._logfile.flush()


class IteratedExperiment(Experiment):
    """This is a an abstract base class, a subclass of :class:`Experiment`, for
    experiements where each participant performs a sequence of identical or similar
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
    is called repeatedly, once for each round, and should return a
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
        ``False`` (or any falsey value) this participant's activities in this condition
        end with no more rounds. The values of *round*, *participant*, *condition* and
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
        describes round this is; it starts at zero and increases by one at each iteration.
        The *participant*, *condition* and *context* are as for :meth:`run_participant`.
        Any changes made to the *context* by :meth:`run_participant_prepare` or by
        previous invocations of :meth:`run_participant_run` or
        :meth:`run_participant_continue` are retained in the *context* presented to this
        method, and any changes this method makes to its *context* are propogated to
        future calls of :meth:`run_participant_continue` and :meth:`run_participant_run`
        by this participant in this condition, but not to any others. The value returned
        by this method, or ``None`` if no value is returne, is collected into a list with
        other return values of this method for other rounds by this participant in this
        condition, which list is eventually passed to the :meth:`run_particiapnt_finish`
        method. This method must be overridden by subclasses, and should not be called
        directly by the programmer. The default implementation of this method raises a
        :exc:`NotImplementedError`.
        """
        raise NotImplementedError("The run_participant_run() method must be overridden")

    def run_participant_finish(self, participant, condition, results):
        """This method is called after all the rounds for a participant in a condition
        have been executed. The *participant* and *condition* are as for
        :meth:`run_participant`. Passed as *results* is a list of the values returned by
        the successive invoations of the :meth:`run_participant_run` method, indexable by
        round. This method should return a `picklable
        <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_ value which
        will be returned to the control process for this participant and condition. This
        method is intended to be overridden in subclasses, and should not be called
        directly by the programmer. The default implementation of this method returns
        *results* unchanged.
        """
        return results

    def run_participant(self, participant, condition, context):
        results = []
        self.run_participant_prepare(participant, condition, context)
        for round in range(self.rounds):
            if not self.run_participant_continue(round, participant, condition, context):
                break
            results.append(self.run_participant_run(round, participant, condition, context))
        self.run_participant_finish(participant, condition, context)
        return self.run_participant_finish(participant, condition, results)
