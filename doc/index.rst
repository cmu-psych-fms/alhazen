Alhazen version 1.1
*******************

.. toctree::
   :maxdepth: 3
   :caption: Contents:


Introduction
============

.. automodule:: alhazen


Installing Alhazen
==================

Alhazen requires Python version 3.7 or later. Recent versions of Mac OS X and recent Linux distributions
are likely to have a suitable version of Python pre-installed, but it may need to be invoked as ``python3``
instead of just ``python``, which latter often runs a 2.x version of Python instead. Use of a virtual environment,
which is recommended, often obviates the need for the ``python3``/``python`` distinction.
If it is not already installed, Python, for Windows, Mac OS X, Linux, or other Unices, can be
`downloaded from python.org <http://www.python.org/download/>`_, for free.

Normally, assuming you are connected to the internet, to install Alhazen you should simply have to type at the command line

  .. parsed-literal:: pip install alhazen

Depending upon various possible variations in how Python and your machine are configured
you may have to modify the above in various ways

* you may need to ensure your virtual environment is activated

* you may need use an alternative scheme your Python IDE supports

* you may need to call it ``pip3`` instead of simply ``pip``

* you may need to precede the call to ``pip`` by ``sudo``

* you may need to use some combination of the above

If you are unable to install Alhazen as above, you can instead
`download a tarball <https://bitbucket.org/dfmorrison/alhazen/downloads/?tab=tags>`_.
The tarball will have a filename something like alhazen-1.1.tar.gz.
Assuming this file is at ``/some/directory/alhazen-1.1.tar.gz`` install it by typing at the command line

  .. parsed-literal:: pip install /some/directory/alhazen-1.1.tar.gz

Alternatively you can untar the tarball with

  .. parsed-literal:: tar -xf /some/directory/alhazen-1.1.tar.gz

and then change to the resulting directory and type

  .. parsed-literal:: python setup.py install


Mailing List
============

There is a `mailing list <https://lists.andrew.cmu.edu/mailman/listinfo/alhazen-users>`_ for those interested in Alhazen and its development.


Tutorial
========

As an example we will model a player performing an iterated, binary
choice task, where one of the two available choices is “safe,” always
earning the player one point, and the other “risky,” sometimes earning
the player ten points, but more frequently none, but with the
probability of the high payoff set to 0.1 so that the expected value
of either choice is one point. The player has no *a priori* knowledge
of the game, and learns of it from their experience. We’ll have 10,000
virtual participants perform this task, each for 200 rounds, learning
from their past experiences in earlier rounds. Then we’ll graph the
average number of times a participant made the risky choice as a
function of the round.

First we start by make a subclass of the Alhazen :class:`Experiment` class,
and add a ``results`` slot to this object that we will use to record
the fraction of participants making the risky choice.

.. code-block:: python

    from alhazen import Experiment

    class SafeRisky(Experiment):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = [0] * self.rounds


The players actions will be decided by an
`Instance Based Learning <https://www.sciencedirect.com/science/article/abs/pii/S0364021303000314>`_
model, written using `PyACTUp <https://halle.psy.cmu.edu/pyactup/>`_.
We will override the :meth:`run_participant` method of our
:class:`Experiment` class. This method will be called in a worker
process for each participant being run by that worker process. For
this first example we will ignore the *participant*, *condition*
and *context* parameters to this method. Note that the return value
from :meth:`run_participant` is delivered to the parent, control
process for aggregation with the results for other participants, both
from the same worker process and from other worker processes. This
return value must be
`picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.


.. code-block:: python

    from alhazen import Experiment
    import pyactup
    import random

    class SafeRisky(Experiment):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = [0] * self.rounds

        def run_participant(self, participant, condition, context):
            risky_chosen = [True] * self.rounds
            mem = pyactup.Memory()
            mem.learn(choice="safe", payoff=12, advance=0)
            mem.learn(choice="risky", payoff=12)
            for r in range(self.rounds):
                choice, bv = mem.best_blend("payoff",
                                            ("safe", "risky"),
                                            "choice")
                if choice == "safe":
                    payoff = 1
                    risky_chosen[r] = False
                elif random.random() < 0.1:
                    payoff = 10
                else:
                    payoff = 0
                mem.learn(choice=choice, payoff=payoff)
            return risky_chosen


Next we override the :meth:`finish_participant` method. This is run in the
parent, control process, and will aggregate the results from the
various worker processes. It is called once for each execution of the
:meth:`run_participant` method in a worker process, with the result
returned from that :meth:`run_participant` execution as the value of the
*result* parameter passed to :meth:`finish_participant`. Again, we are
currently ignoring the *participant* and *condition* parameters.

.. code-block:: python

    from alhazen import Experiment
    from itertools import count
    import pyactup
    import random

    class SafeRisky(Experiment):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = [0] * self.rounds

        def run_participant(self, participant, condition, context):
            risky_chosen = [True] * self.rounds
            mem = pyactup.Memory()
            mem.learn(choice="safe", payoff=12, advance=0)
            mem.learn(choice="risky", payoff=12)
            for r in range(self.rounds):
                choice, bv = mem.best_blend("payoff",
                                            ("safe", "risky"),
                                            "choice")
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


This all that is required to be able to run our model in multiple
processes. The :class:`Experiment` super-class will handle the partitioning
of participants across worker processes, tell them run, collect their
results, all while correctly transferring information between the
various processes and safely synchronizing their activities. When
finished the final, aggregated result will be available in the
:class:*SafeRisky*’s *results* slot.

To make use of this we will add a :func:`main` function that will allocate
a :class:`SafeRisky` object, initialized with the desired number of rounds
and participants, call its :meth:`run` method, and draw a graph of the
results with Matplotlib. The resulting program can also be called from
the command line with an optional parameter specifying the number of
worker processes to use.

.. code-block:: python

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

        def run_participant(self, participant, condition, context):
            risky_chosen = [True] * self.rounds
            mem = pyactup.Memory()
            mem.learn(choice="safe", payoff=12, advance=0)
            mem.learn(choice="risky", payoff=12)
            for r in range(self.rounds):
                choice, bv = mem.best_blend("payoff",
                                            ("safe", "risky"),
                                            "choice")
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
                  help=("number of worker processes, zero (the default) "
                        "means as many as available cores"))
    def main(**kwargs):
        sr = SafeRisky(participants=PARTICIPANTS,
                       rounds=ROUNDS,
                       process_count=kwargs["workers"])
        sr.run()
        plt.plot(range(1, sr.rounds + 1), sr.results)
        plt.xlabel("round")
        plt.ylim(-0.05, 1.05)
        plt.ylabel("fraction choosing risky")
        plt.title(f"Safe versus Risky, {sr.participants:,d} participants")
        plt.show()


    if __name__== "__main__":
        main()


When run a graph like the following is displayed. We see that the model has
a strong bias against the risky choice, even though it has the same expected
value as the safe one.

    .. image:: simple.png

Using a particular 32 core machine, running this with ``--workers=1``,
to use just a single worker process, it requires one minute and
thirty-nine seconds to run to completion. If instead it is run with
``--workers=32``, to use 32 worker processes, all able to run in
parallel on the 32 core machine, it completes in only three seconds.

Often we want to run experiments like these with multiple, different
conditions, such as different parameters to the models or to the
tasks. This is facilitated by using the *conditions* slot of the
:class:`Experiment` object, an iterable the elements of which are passed to
the :meth:`run_participant` method in the work process. Note that the
total number of participants is the product of *participants* and
the number of conditions. A condition can be any
`picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_
Python value, though it is often most convenient to make it also a hashable
value so that data can be gather into dictionaries indexed by
conditions. Note that it is easy to run cross products of two or more
orthogonal sets of conditions by using tuples of their elements.

Here we augment the above :class:`SafeRisky` implementation to use different
probabilities for the risky choice, for different expected values of that
choice. The values we pass as conditions will be numbers, the desired expected
value.

.. code-block:: python

    from alhazen import Experiment
    import click
    from itertools import count
    import matplotlib.pyplot as plt
    import pyactup
    import random

    ROUNDS = 200
    PARTICIPANTS = 10_000
    EXPECTED_VALUES = [5, 4, 3, 2, 1]

    class SafeRisky(Experiment):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = {c: [0] * self.rounds for c in self.conditions}

        def run_participant(self, participant, condition, context):
            p = condition / 10
            risky_chosen = [True] * self.rounds
            mem = pyactup.Memory()
            mem.learn(choice="safe", payoff=12, advance=0)
            mem.learn(choice="risky", payoff=12)
            for r in range(self.rounds):
                choice, blended_value = mem.best_blend("payoff",
                                                       ("safe", "risky"),
                                                       "choice")
                if choice == "safe":
                    payoff = 1
                    risky_chosen[r] = False
                elif random.random() < p:
                    payoff = 10
                else:
                    payoff = 0
                mem.learn(choice=choice, payoff=payoff)
            return risky_chosen

        def finish_participant(self, participant, condition, result):
            for r, i in zip(result, count()):
                if r:
                    self.results[condition][i] += r / self.participants


    @click.command()
    @click.option("--workers", "-w", default=0,
                  help=("number of worker processes, zero (the default) "
                        "means as many as available cores"))
    def main(**kwargs):
        sr = SafeRisky(participants=PARTICIPANTS,
                       rounds=ROUNDS,
                       conditions=EXPECTED_VALUES,
                       process_count=kwargs["workers"])
        sr.run()
        for c in sr.conditions:
            plt.plot(range(1, sr.rounds + 1), sr.results[c],
                     label=f"Risky EV = {c}")
        plt.legend()
        plt.xlabel("round")
        plt.ylim(-0.05, 1.05)
        plt.ylabel("fraction choosing risky")
        plt.title(f"Safe versus Risky, {sr.participants:,d} participants")
        plt.show()


    if __name__== "__main__":
        main()


When run, a graph like the following is displayed. We see that the
probability of getting the high payoff must be surprisingly high to
convince the model to overcome its averseness to risk.

    .. image:: conditions.png

Again on a 32 core machine, running this with ``--workers=1``, it
requires seven minutes and forty-three seconds, but with multiple, parallel worker
processes,``--workers=32``, only fifteen seconds.

See details of the API in the next section for other methods that can
be overridden to hook into different parts of the process of running
the experiment.



API Reference
=============

.. autoclass:: Experiment

   .. autoattribute:: participants

   .. autoattribute:: conditions

   .. autoattribute:: rounds

   .. autoattribute:: process_count

   .. autoattribute:: show_progress

   .. automethod:: run

   .. automethod:: run_participant

   .. automethod:: finish_participant

   .. automethod:: prepare_condition

   .. automethod:: prepare_experiment

   .. automethod:: setup

   .. automethod:: finish_experiment

