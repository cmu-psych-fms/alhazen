Alhazen version 1.3.2
*********************

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
The tarball will have a filename something like alhazen-1.3.1.tar.gz.
Assuming this file is at ``/some/directory/alhazen-1.3.1.tar.gz`` install it by typing at the command line

  .. parsed-literal:: pip install /some/directory/alhazen-1.3.1.tar.gz

Alternatively you can untar the tarball with

  .. parsed-literal:: tar -xf /some/directory/alhazen-1.3.1.tar.gz

and then change to the resulting directory and type

  .. parsed-literal:: python setup.py install


Mailing List
============

There is a `mailing list <https://lists.andrew.cmu.edu/mailman/listinfo/alhazen-users>`_ for those interested in Alhazen and its development.


Tutorial
========

As an example we will model a player performing an iterated, binary
choice task, where one of the two available choices is “safe,” always
earning the player one point, and the other is “risky,” sometimes earning
the player ten points, but more frequently zero, but with the
probability of the high payoff set to 0.1 so that the expected value
of either choice is one point. The player has no *a priori* knowledge
of the game, and learns of it from experience. Some number of
virtual participants will perform this task for a fixed number of
rounds, learning from their past experiences in earlier rounds. Then
we’ll graph the average number of times a participant made the risky
choice as a function of the round. In our implementation the simulated
participants’ actions will be decided by an
`Instance Based Learning <https://www.sciencedirect.com/science/article/abs/pii/S0364021303000314>`_
model, written using `PyACTUp <https://halle.psy.cmu.edu/pyactup/>`_.

First we start by make a subclass of the Alhazen
:class:`IteratedExperiment` class, and override its
:meth:`run_participant_prepare` method to allocate for each
participant a PyACTUp ``Memory`` object. This method will be called
within a worker process, and for this example we will ignore the
*participant*, *condition* and *context* parameters to this method.

.. code-block:: python

    from alhazen import Experiment
    import pyactup

    class SafeRisky(IteratedExperiment):

        def run_participant_prepare(self, participant, condition, context):
            self.memory = pyactup.Memory()

We next initialize the PyACTUp ``Memory`` and override the
:meth:`run_participant_run` method to actually implement the cognitive
model. It makes a choice, learns the payoff that that choice produced,
and returns whether or not it made the risky choice, for subsequent
reporting by the parent, control process. This return value must be
`picklable
<https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_.
Again this method is called in a worker process, and we are ignoring
the values of the *round*, *participant*, *condition* and *context*
parameters.

.. code-block:: python

    from alhazen import IteratedExperiment
    import pyactup
    import random

    class SafeRisky(IteratedExperiment):

        def run_participant_prepare(self, participant, condition, context):
            self.memory = pyactup.Memory()
            self.memory.learn(choice="safe", payoff=12, advance=0)
            self.memory.learn(choice="risky", payoff=12)

        def run_participant_run(self, round, participant, condition, context):
            choice, bv = self.memory.best_blend("payoff", ("safe", "risky"), "choice")
            if choice == "safe":
                payoff = 1
            elif random.random() < 0.1:
                payoff = 10
            else:
                payoff = 0
            self.memory.learn(choice=choice, payoff=payoff)
            return choice == "risky"

This all that is required to be able to run our model in multiple
processes. The :class:`IteratedExperiment` super-class will handle the partitioning
of participants across worker processes, tell them to run, collect their
results, all while correctly transferring information between the
various processes and safely synchronizing their activities. The collected
results will be available as the value returned by the :class:`SafeRisky`'s
:meth:`run` method.

To make use of this we will add a :func:`main` function that will
allocate a :class:`SafeRisky` object, initialized with the desired
number of rounds and participants, call its :meth:`run` method, and
draw a graph of the results with Matplotlib. This progream takes three
command line arguments to specify the number of participants, number
of rounds and number of worker processes. If these arguments are not
provided explicitly they default to 10,000 participants, 200 rounds,
and as many worker processes as the machine it is running in has
cores.

.. code-block:: python

    from alhazen import IteratedExperiment
    import click
    import matplotlib.pyplot as plt
    import pyactup
    import random

    class SafeRisky(IteratedExperiment):

        def run_participant_prepare(self, participant, condition, context):
            self.memory = pyactup.Memory()
            self.memory.learn(choice="safe", payoff=12, advance=0)
            self.memory.learn(choice="risky", payoff=12)

        def run_participant_run(self, round, participant, condition, context):
            choice, bv = self.memory.best_blend("payoff", ("safe", "risky"), "choice")
            if choice == "safe":
                payoff = 1
            elif random.random() < 0.1:
                payoff = 10
            else:
                payoff = 0
            self.memory.learn(choice=choice, payoff=payoff)
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

When run with the default command line arguments a graph like the
following is displayed. We see that the model has a strong bias
against the risky choice, even though it has the same expected value
as the safe one.

    .. image:: simple.png

Using a particular 32 core machine, running this with ``--workers=1``,
to use just a single worker process, it requires one minute and
thirty-nine seconds to run to completion. If instead it is run with
``--workers=32``, to use 32 worker processes, all able to run in
parallel on the 32 core machine, it completes in only three seconds.

Often we want to run experiments like these with multiple, different
conditions, such as different parameters to the models or to the
tasks. This is facilitated by using the *conditions* slot of the
:class:`IteratedExperiment` object, an iterable the elements of which
are passed to the :meth:`run_participant` method in the worker process.
Note that the total number of participants is the product of
*participants* and the number of conditions. A condition can be any
Python value that is both hashable and
`picklable <https://docs.python.org/3.7/library/pickle.html#pickle-picklable>`_,
Note that it is easy to run cross products of two or more orthogonal
sets of conditions by using tuples of their elements.

Here we augment the above :class:`SafeRisky` implementation to use different
probabilities for the risky choice, for different expected values of that
choice. The values we pass as conditions will be numbers, the desired expected
values. At the same time we enable recording the individual choices made at each round by
each participant in a comma separated values (CSV) file, write access to which
is synchronized across the worker processes using the :attr:`log` property.

.. code-block:: python

    from alhazen import IteratedExperiment
    import click
    from itertools import count
    import matplotlib.pyplot as plt
    import pyactup
    import random

    EXPECTED_VALUES = [5, 4, 3, 2, 1]

    class SafeRisky(IteratedExperiment):

        def prepare_experiment(self, **kwargs):
            with self.log as w:
                if w:
                    w.writerow("expected value,participant,round,choice,payoff".split(","))

        def run_participant_prepare(self, participant, condition, context):
            self.memory = pyactup.Memory()
            self.memory.learn(choice="safe", payoff=12, advance=0)
            self.memory.learn(choice="risky", payoff=12)

        def run_participant_run(self, round, participant, condition, context):
            choice, bv = self.memory.best_blend("payoff", ("safe", "risky"), "choice")
            if choice == "safe":
                payoff = 1
            elif random.random() < condition / 10:
                payoff = 10
            else:
                payoff = 0
            self.memory.learn(choice=choice, payoff=payoff)
            with self.log as w:
                if w:
                    w.writerow([condition, participant, round, choice, payoff])
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
                        csv=True)
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


When run with the default command line arguments, a graph like the
following is displayed. We see that the probability of getting the
high payoff must be surprisingly high to convince the model to
overcome its distaste for risk.

    .. image:: conditions.png

Again on a 32 core machine, running this with ``--workers=1``, it
requires seven minutes and forty-three seconds, but with multiple, parallel worker
processes,``--workers=32``, only fifteen seconds.

If in addition we specify a log file when running this, the first few lines of that
log file look something like the following.

.. code-block::

    expected value,participant,round,choice,payoff
    5,3,0,safe,1
    5,3,1,risky,0
    5,3,2,safe,1
    5,3,3,safe,1
    5,3,4,risky,10
    5,3,5,risky,0
    5,3,6,safe,1
    5,3,7,risky,10
    5,3,8,risky,10
    5,3,9,risky,10


See details of the API in the next section for other methods that can
be overridden to hook into different parts of the process of running
the experiment, as well as for the underlying :meth:`Experiment`
class.



API Reference
=============

Experiments
-----------

.. autoclass:: Experiment

   .. autoattribute:: participants

   .. autoattribute:: conditions

   .. autoattribute:: process_count

   .. autoattribute:: show_progress

   .. automethod:: run

   .. automethod:: run_participant

   .. automethod:: finish_participant

   .. automethod:: prepare_condition

   .. automethod:: finish_condition

   .. automethod:: prepare_experiment

   .. automethod:: setup

   .. automethod:: finish_experiment

   .. autoattribute:: log

Iterated Experiments
--------------------

.. autoclass:: IteratedExperiment

   .. autoattribute:: rounds

   .. automethod:: run_participant_prepare

   .. automethod:: run_participant_run

   .. automethod:: run_participant_continue

   .. automethod:: run_participant_finish
