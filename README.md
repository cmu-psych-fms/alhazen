Alhazen is a small, simple framework to facilitate running
experiments, written in Python, using cognitive models, or similar
applications, in multiple, parallel processes. It is primarily useful on
multi-core machines, though most modern machines are such; the more
cores, the more performance benefit you are likely to get by using it.
It also depends upon the experiment being structured as a large number
of identical, independent runs of the same activity, or of similar
activities. This is a common pattern, each such run usually
corresponding to a distinct virtual participant, or possibly a
collection of interacting participants.

There is [online documentation of Alhazen](https://halle.psy.cmu.edu/alhazen/),
and the [sources](https://github.com/cmu-psych-fms/alhazen) are on GitHub.


Alhazen requires Python version 3.7 or later. Recent versions of Mac OS X and recent Linux distributions
are likely to have a suitable version of Python pre-installed, but it may need to be invoked as ``python3``
instead of just ``python``, which latter often runs a 2.x version of Python instead. Use of a virtual environment,
which is recommended, often obviates the need for the ``python3``/``python`` distinction.
If it is not already installed, Python, for Windows, Mac OS X, Linux, or other Unices, can be
`downloaded from python.org <http://www.python.org/download/>`_, for free.

Normally, assuming you are connected to the internet, to install Alhazen you should simply have to type at the command line

    pip install alhazen

Depending upon various possible variations in how Python and your machine are configured
you may have to modify the above in various ways

* you may need to ensure your virtual environment is activated

* you may need use an alternative scheme your Python IDE supports

* you may need to call it ``pip3`` instead of simply ``pip``

* you may need to precede the call to ``pip`` by ``sudo``

* you may need to use some combination of the above

Alhazen is released under the following MIT style license:

Copyright (c) 2020-2025 Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
