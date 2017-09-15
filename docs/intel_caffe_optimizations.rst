.. _caffe-optimization: 

.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel Corporation
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

*****************************************
Optimizing Intel® distribution of Caffe*
*****************************************

.. Note::
   [As this is a rough draft, some of the existing source documentation I've compiled here might need to be copy edited/branding edited/formatted once we fill this article out.]
   Describe configuration for a common standard platform - Xeon Skylake, possible additional documentation for other platforms if necessary. 
   Contact Caffe team for new information
   BKMs for other frameworks. 

Release notes
=============

* 0.1 - Initial draft of Caffe performance optimization for Intel Architecture
* 0.2 - Review draft for Caffe team

Introduction
============

.. Note::
   [In this section, we want to briefly describe the purpose, goals, and audience for this article.]

Caffe* is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (`BVLC <http://bvlc.eecs.berkeley.edu/>`_) and by community contributors. `Yangqing Jia <http://daggerfs.com/>`_ created the project during his PhD at UC Berkeley. Caffe is released under the `BSD 2-Clause license <https://github.com/BVLC/caffe/blob/master/LICENSE/>`_.

These instructions walk throught the configuration steps needed to optimize the Intel® distribution of Caffe* on platforms based on Intel® Xeon® processors, code named Skylake. 

.. Note:: 
   We need an actual processor number here to serve as our reference platform. Skylake processors are: processor family E3-12xx v5 and E3-12xx v5 processors, Xeon E3-12xx v5  Xeon W-21xx, Xeon Bronze, Silver, Gold, Platinum.

[[Additional platforms supported are Intel Xeon processors and Intel Xeon Phi processors code named:
* Broadwell
* Skylake
* Knights Mill
* Knights Landing]]


Prerequisites
=============

.. Note::
   [This section needs to describe the prerequisites for each supported environment. Any additional packages, libraries, etc. We probably do need to update, copy edit, and link to these existing documents]


The prerequisites for the Intel distribution of Caffe differ depending on the target system's OS. 

[[these OS-dependent sections are best if they are separate files that we link to in this document (install_yum, install_apt, and install_osx). However, because these procedures are old and need to be reviewed, I've copied the contents of these files below so we can update them as necessary. ]]

Install on OSX*
---------------

We highly recommend using the Homebrew package manager. Ideally, start from a clean */usr/local* to avoid conflicts. In the following instructions, we assume that you're using `Anaconda Python <https://docs.anaconda.com/anaconda/>`_ and `Homebrew <https://brew.sh/>`_.

CUDA
^^^^

Install via the NVIDIA package that includes both CUDA and the bundled driver. We strongly recommend using CUDA 7. Older CUDA versions require *libstdc++*, while *clang++* is the default compiler and *libc++* is the default standard library on OSX* 10.9+. This conflict makes it necessary to change the compilation settings for each of the dependencies, which is prone to error and is not recommended.

Library Path
^^^^^^^^^^^^

We find that everything compiles successfully if *$LD_LIBRARY_PATH* is not set at all, and *$DYLD_FALLBACK_LIBRARY_PATH* is set to provide CUDA, Python, and other relevant libraries (For example: */usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib*). In other ENV settings, things might not work as expected.

General dependencies
~~~~~~~~~~~~~~~~~~~~

::

  brew install -vd snappy leveldb gflags glog szip lmdb
  # need the homebrew science source for OpenCV and hdf5
  brew tap homebrew/science
  brew install hdf5 opencv

If using Anaconda Python, you might need to modify the OpenCV* formula. 

Do ``brew edit opencv`` and replace the lines that look similar to the two lines below with exactly the two lines below::

  -DPYTHON_LIBRARY=#{py_prefix}/lib/libpython2.7.dylib
  -DPYTHON_INCLUDE_DIR=#{py_prefix}/include/python2.7

HDF5 is bundled with Anaconda Python, so if you are using Anaconda Python, you can skip the hdf5 formula.

Remaining dependencies, with / without Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  # with Python pycaffe needs dependencies built from source
  brew install --build-from-source --with-python -vd protobuf
  brew install --build-from-source -vd boost boost-python
  # without Python the usual installation suffices
  brew install protobuf boost

**BLAS**: already installed as the Accelerate / vecLib framework. OpenBLAS* and Intel MKL are alternatives for faster CPU computation.

**Python** (optional): Anaconda is the preferred Python distribution. If you decide against using it, please use Homebrew. Check that Caffe and dependencies are linking against the same, desired Python distribution.

Continue with compilation.


libstdc++ installation
^^^^^^^^^^^^^^^^^^^^^^

This route is not for the faint of heart. For OSX 10.10 and 10.9 you should install CUDA 7 and follow the instructions in the section above. If that is not an option, take a deep breath and carry on with the instructions in this section.

In OS X 10.9+, *clang++* is the default C++ compiler and uses *libc++* as the standard library. However, NVIDIA* CUDA (even version 6.0) currently links only with *libstdc++*. This makes it necessary to change the compilation settings for each of the dependencies.

We do this by modifying the Homebrew formulae before installing any packages. Make sure that Homebrew doesn't install any software dependencies in the background; all packages must be linked to *libstdc++*.

The prerequisite Homebrew formulae are::

  boost snappy leveldb protobuf gflags glog szip lmdb homebrew/science/opencv

For each of these formulas, ``brew edit FORMULA``, and then add the ENV definitions as shown::

  def install
      # ADD THE FOLLOWING:
      ENV.append "CXXFLAGS", "-stdlib=libstdc++"
      ENV.append "CFLAGS", "-stdlib=libstdc++"
      ENV.append "LDFLAGS", "-stdlib=libstdc++ -lstdc++"
      # The following is necessary because libtool likes to strip LDFLAGS:
      ENV["CXX"] = "/usr/bin/clang++ -stdlib=libstdc++"
      ...

To edit the formulae in turn, run::

  for x in snappy leveldb protobuf gflags glog szip boost boost-python lmdb homebrew/science/opencv; do brew edit $x; 

After this, run::

  for x in snappy leveldb gflags glog szip lmdb homebrew/science/opencv; do brew uninstall $x; brew install --build-from-source -vd $x; done
  brew uninstall protobuf; brew install --build-from-source --with-python -vd protobuf
  brew install --build-from-source -vd boost boost-python

If these commands are not executed exactly right then linking errors will trouble you.

The Homebrew versioning that Homebrew maintains itself as a separate Git* repository, and making the above ``brew edit FORMULA`` changes will change files in your local copy of Homebrew's master branch. By default, this will prevent you from updating Homebrew using ``brew update``, as you will get an error message like the following::

  $ brew update
  error: Your local changes to the following files would be overwritten by merge:
    Library/Formula/lmdb.rb
  Please, commit your changes or stash them before you can merge.
  Aborting
  Error: Failure while executing: git pull -q origin refs/heads/master:refs/remotes/origin/master

One solution to this is to commit your changes to a separate Homebrew branch, run ``brew update``, and then rebase your changes onto the updated master. You'll have to do this both for the main Homebrew repository in */usr/local/* and also the Homebrew science repository that contains OpenCV in */usr/local/Library/Taps/homebrew/homebrew-science*, as follows::

  cd /usr/local
  git checkout -b caffe
  git add .
  git commit -m "Update Caffe dependencies to use libstdc++"
  cd /usr/local/Library/Taps/homebrew/homebrew-science
  git checkout -b caffe
  git add .
  git commit -m "Update Caffe dependencies"

Then, whenever you want to update Homebrew, you must switch back to the master branches, do the update, rebase the Caffe branches onto the master branch, and then fix any conflicts, as shown below::

  # Switch batch to homebrew master branches
  cd /usr/local
  git checkout master
  cd /usr/local/Library/Taps/homebrew/homebrew-science
  git checkout master

::

  # Update homebrew; hopefully this works without errors!
  brew update

::

  # Switch back to the caffe branches with the formulae that you modified earlier
  cd /usr/local
  git rebase master caffe
  # Fix any merge conflicts and commit to caffe branch
  cd /usr/local/Library/Taps/homebrew/homebrew-science
  git rebase master caffe
  # Fix any merge conflicts and commit to caffe branch
  # Done!

At this point, you should be running the latest Homebrew packages and your Caffe-related modifications will remain in place.


Ubuntu installation 
-------------------

General dependencies
^^^^^^^^^^^^^^^^^^^^

::

  sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
  sudo apt-get install --no-install-recommends libboost-all-dev

**CUDA**: Install via the NVIDIA package instead of with ``apt-get`` to be certain of the library and driver versions. Install the library and latest driver separately; the driver bundled with the library is usually out-of-date. This can be skipped for CPU-only installation.

**BLAS**: Install ATLAS using ``sudo apt-get install libatlas-base-dev``, or install OpenBLAS or Intel MKL for better CPU performance.

**Python** (optional): if you use the default Python distribution you will need to run ``sudo apt-get install`` for the ``python-dev <package>`` to have the Python headers for building the *pycaffe* interface.

Remaining dependencies, 14.04
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything is packaged in 14.04.

::

  sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

Remaining dependencies, 12.04
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These dependencies need manual installation in 12.04.

::

  # glog
  wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
  tar zxvf glog-0.3.3.tar.gz
  cd glog-0.3.3
  ./configure
  make && make install
  # gflags
  wget https://github.com/schuhschuh/gflags/archive/master.zip
  unzip master.zip
  cd gflags-master
  mkdir build && cd build
  export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
  make && make install
  # lmdb
  git clone https://github.com/LMDB/lmdb
  cd lmdb/libraries/liblmdb
  make && make install

Note that ``glog`` does not compile with the most recent *gflags* version (2.1), so before that is resolved you will need to build with ``glog`` first.

Continue with compilation.


RHEL / Fedora / CentOS Installation
-----------------------------------

General dependencies
^^^^^^^^^^^^^^^^^^^^

::

  sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel

Remaining dependencies, recent OS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  sudo yum install gflags-devel glog-devel lmdb-devel

Remaining dependencies, if not found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  # glog
  wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
  tar zxvf glog-0.3.3.tar.gz
  cd glog-0.3.3
  ./configure
  make && make install
  # gflags
  wget https://github.com/schuhschuh/gflags/archive/master.zip
  unzip master.zip
  cd gflags-master
  mkdir build && cd build
  export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
  make && make install
  # lmdb
  git clone https://github.com/LMDB/lmdb
  cd lmdb/libraries/liblmdb
  make && make install

Note that ``glog`` does not compile with the most recent *gflags* version (2.1), so before that is resolved you will need to build with glog first.

**CUDA**: Install CUDA via the NVIDIA package instead of with ``yum`` to be certain of the library and driver versions. Install the library and latest driver separately; the driver bundled with the library is usually out-of-date for CentOS/RHEL/Fedora.

**BLAS**: Install ATLAS using ``sudo yum install atlas-devel``, or install OpenBLAS or Intel MKL for better CPU performance. For the Makefile build, uncomment and set ``BLAS_LIB`` accordingly, as ATLAS is usually installed under */usr/lib[64]/atlas)*.

**Python** (optional): If you use the default Python you will need to ``sudo yum install`` the python-devel package to have the Python headers for building the pycaffe wrapper.

Then continue with compilation.


Installing Intel MKL-DNN
------------------------

.. _Note: 
   This section follows the linked OS-dependent dependencies above. We might want to link out to the MKL-DNN and MKL installation instructions instead of duplicating them here.

.. _Note: 
   Intel offers users the choice of using either `Intel MKL-DNN <https://github.com/intel/mklnn/>`_ for developers looking for an open source performance library for Deep Learning applications, or `Intel MKL <https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2017-install-guide/>`_ for developers who want a Intel-proprietary computing math library for applications that require maximum performance.

This section elaborates on the installation information presented on the `GitHub repository site <https://github.com/intel/mklnn/>`_ by providing detailed, step-by-step instructions for installing and building the Intel MKL-DNN library components. The computer you use require an Intel® processor supporting Intel® Advanced Vector Extensions 2 (Intel® AVX2). Specifically, Intel MKL-DNN is optimized for Intel® Xeon® processors, Intel® Xeon Phi™ processors, and `Intel AVX-512 <https://www.intel.com/content/www/us/en/architecture-and-technology/avx-512-overview.html/>`_.

GitHub indicates the software was validated on RedHat* Enterprise Linux* 7; however, the information presented in this tutorial was developed on a system running Ubuntu* 16.04.

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

Intel MKL-DNN has the following dependencies:

  * CMake* – a cross-platform tool used to build, test, and package software.
  * Doxygen* – a tool used to generate documentation from annotated source code.

If these software tools are not already installed on your computer, you can install them with the following commands::

  sudo apt install cmake
  sudo apt install doxygen

.. _Note: 
   If Git* is not already set up on your computer, you can install it by typing the following::

    sudo apt install git

Download and Build the Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the Intel MKL-DNN library from the GitHub repository by opening a terminal and typing the following command::

  git clone https://github.com/01org/mkl-dnn.git

Once the installation has completed you will find a directory named *mkl-dnn* in the Home directory. Navigate to this directory by typing::

  cd mkl-dnn

As explained on the GitHub repository site, Intel MKL-DNN uses the optimized general matrix to matrix multiplication (GEMM) function from Intel MKL. The library supporting this function is also included in *mkl-dnn* repository. You can download the library by running the *prepare_mkl.sh* script located in the scripts directory::

  cd scripts && ./prepare_mkl.sh && cd ..

This script creates a directory named *external* and then downloads and extracts the library files to a directory named *mkl-dnn/external/mklml_lnx*.

Execute the next command from the *mkl-dnn* directory. This command creates a subdirectory named *build* and then runs CMake and Make to generate the build system::

  mkdir -p build && cd build && cmake .. && make

Use the automated script to install additional libraries, list folders, etc.

Other frameworks will require more description.


Building for Intel® Architecture
================================

.. _Note: 
   Check with Caffe team to verify this procedure. Boost and GEM isn't optimized; works best with MKL and MKLDNN. Frank y zhang and Daisy Deng to find out how to build MKL and MKL-DNN

https://github.com/intel/caffe/blob/master/docs/release_notes.md#Building

This version of Caffe is optimized for Intel® Xeon processors and Intel® Xeon Phi™ processors. To achieve the best performance results on Intel architecture we recommend building the Intel® Distribution of Caffe* with Intel® MKL and enabling OpenMP* support. This Caffe version is self-contained. This means that newest version of Intel MKL will be downloaded and installed during compilation of Intel® Distribution of Caffe*.

Set ``BLAS := mkl`` in ``Makefile.config``

If you don't need GPU optimizations, set the ``CPU_ONLY := 1`` flag in ``Makefile.config`` to configure and build the Intel® Distribution of Caffe* without CUDA.

Intel MKL 2017 introduces optimized Deep Neural Network (DNN) performance primitives that accelerate the most popular image recognition topologies. The Intel® Distribution of Caffe* can take advantage of these primitives and get significantly better performance results compared to the previous versions of Intel MKL. There are two ways you can take advantage of the new primitives:

* Set layer engine to ``MKL2017`` in the prototxt file (model). Only this specific layer will be accelerated with new primitives.
* Use ``-engine = MKL2017`` in the command line as an option when executing Caffe for training, scoring, or benchmarking.

.. Note::
   {DO WE NEED TO INCLUDE INSTRUCTIONS ON BUILDING FOR GPU?}

Compilation
===========

Caffe can be compiled with either Make or CMake. Make is officially supported, while CMake is supported by the community. The build procedure for both is the same as it is on the *bvlc-caffe-master* branch. When OpenMP is available, it will be used automatically.

Compilation with Make
---------------------

Configure the build by copying and modifying the example Makefile.config for your setup. The defaults should work, but you should uncomment the relevant lines if you are using Anaconda Python.

::

  cp Makefile.config.example Makefile.config
  # Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)

::

  make all
  make test
  make runtest

For both CPU- and GPU-accelerated Caffe, no changes are needed.

For cuDNN acceleration using NVIDIA's proprietary cuDNN software, uncomment the ``USE_CUDNN := 1`` switch in *Makefile.config*. cuDNN is sometimes but not always faster than Caffe's GPU acceleration.

For CPU-only Caffe, uncomment ``CPU_ONLY := 1`` in *Makefile.config*.

To compile the Python and MATLAB wrappers, set your MATLAB and Python paths in *Makefile.config*, and then run ``make pycaffe`` and ``make matcaffe`` respectively. 

**Distribution**: Run ``make distribute`` to create a *distribute* directory with all the Caffe headers, compiled libraries, binaries, etc. needed for distribution to other machines.

**Speed**: For a faster build, compile in parallel by doing ``make all -j8`` where ``8`` is the number of parallel threads for compilation (a good choice for the number of threads is the number of processor cores in your machine).

Now that you have installed Caffe, check out the MNIST tutorial and the reference ImageNet model tutorial.


Compilation with CMake
----------------------

In lieu of manually editing *Makefile.config* to configure the build, Caffe offers an unofficial CMake build thanks to `@Nerei <https://github.com/Nerei/>`_, `@akosiorek <https://github.com/akosiorek/>`_ , and other members of the community. It requires CMake version >= 2.8.7. The basic steps are as follows::

  mkdir build
  cd build
  cmake ..
  make all
  make install
  make runtest

See `PR#1667 <https://github.com/BVLC/caffe/pull/1667/>`_ for options and details.


Building with the Intel Compiler
--------------------------------

.. Note::
   [Please confirm that these steps are complete and accurate as of the most current versions of Caffe and Nervana.]

Builing the Intel Distribution of Caffe with the Intel® Compiler allows you to take full advantage of the Intel® processor. This is a step-by-step tutorial for building the Intel Distribution of Caffe with the Intel® MKL-DNN library.

1. Build the Boost library.

    Download the Boost library from the `offical page <http://www.boost.org/>`_ and unzip it.

    Execute the following commands in the order shown::

        Run source <compiler root>/bin/compilervars.sh {ia32 OR intel64} or source <compiler root>/bin/compilervars.csh {ia32 OR intel64}
        cd <boost root>
        ./bootstrap.sh
        ./b2 install --prefix=<Boost.Build install dir>
    
    For 32-bit systems::

        ./b2 --build-dir=<Boost object directory> toolset=intel stage

    For 64-bit systems::

        ./b2 --build-dir=<Boost object directory> address-model=64 toolset=intel stage

2. Update Caffe's code so it is supported by the Intel Compiler.

    We need to add the ``-xHost`` flag to the compiler flag settings for better performance on Intel processors:

        * Add the ``-xHost`` to the variable ``CXX_HARDENING_FLAGS`` on line 373 of */path/to/caffe/Makefile*.
        * Add the ``-xHost`` to the variable ``COMMON_FLAGS`` on line 428 of */path/to/caffe/Makefile*.
        * Modifiy lines 46 and 53 to ``$(eval CXXFLAGS += -DMKLDNN_SUPPORTED -xHost)`` of */path/to/caffe/Makefile.mkldnn*.

3. Build Caffe:

    * cd to */path/to/caffe* and create the *Makefile.config* from the *Makefile.config.example*.
    * Set the variable ``CUSTOM_CXX`` to ``/path/to/icpc.`` For example: ``CUSTOM_CXX := /opt/intel/compilers_and_libraries/linux/bin/intel64/icpc``.
    * Set the variable ``BOOST_ROOT`` to ``/path/to/unzipped_boost_root.`` For example: ``BOOST_ROOT := /home/user/boost_1_64_0``.
    * Run ``make all -j$(nproc)`` to build the Intel Distribution of Caffe.


Configuration
=============

.. Note::
   [Please confirm the instructions below are correct and complete.]

To achieve the best performance with the Intel® distribution of Caffe* on Intel processors, apply the configuration recommendations in this section.

Hardware / BIOS configuration
-----------------------------

* Make sure that your hardware configurations include a fast SSD (M.2) drive. If during training/scoring you observe a "waiting for data" message in the logs, you should install a better SSD or reduce your batchsize.
* For systems based on the Intel Xeon Phi™ product family:
    * Enter BIOS (MCDRAM section) and set MCDRAM mode as cache.
    * Enable Intel® Hyper-Threading Technology (Intel® HT Technology) on your platform. Intel® HT Technology settings can be found in the CPU section of your system BIOS.
* Optimize your system hardware in BIOS: 
    * Set the CPU to max frequency.
    * Set fan speed to 100%.
    * Check the cooling system.
* For multinode systems based on the Intel Xeon Phi™ product family over Intel® Omni-Path Architecture use the following settings:
    * Processor C6 = Enabled
    * Snoop Holdoff Count = 9
    * Intel Turbo Boost Technology = Enabled
    * Uncore settings: Cluster Mode: All2All

Software / OS configuration
---------------------------

For systems based on the Intel® Xeon Phi™ product family:

* We recommend using Linux CentOS 7.2 or newer.
* We recommend using the newest XPPSL software for the Intel Xeon Phi™ product family. 
    * https://software.intel.com/en-us/articles/xeon-phi-software#downloads
    * https://software.intel.com/en-us/articles/xeon-phi-software#downloads

* For multinode systems based on the Intel Xeon Phi™ product family over Intel® Omni-Path Architecture:
    * *irqbalance* needs to be installed and configured with the ``--hintpolicy=exact`` option enabled.
    * CPU frequency needs to be set via the ``intel_pstate`` driver using the following commands::

          echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct
          echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
          cpupower frequency-set -g performance```

Make sure that there are no unnecessary processes running during training and scoring. Intel® Distribution of Caffe* uses all available system resources, so other running processes (like monitoring tools, Java* processes, network traffic, etc.) might impact performance.

We recommend compiling the Intel® Distribution of Caffe* with GCC 4.8.5 (or newer)

We recommend compiling the Intel® Distribution of Caffe* with the following settings in *makefile.configuration* set to::

    CPU_ONLY := 1
    BLAS := mkl

Intel® Distribution of Caffe / Hyper-Parameters configuration*
--------------------------------------------------------------

[These are examples. Say that you can find examples in models/IA-optimized-models folder, don't duplicate that information here. Links here, remove text from this section]

We provide two sets of prototxt files with Hyper-Parameters and network topologies. In the default set you will find standard topologies and the configurations used by the community. In BKM (Best Know Method) you will find our internally-developed solution that is optimized for Intel MKL2017 and Intel processors.

When running performance and training, we recommend that you start working with the default sets to establish a performance baseline.

Use the LMDB data layer (Using the *Images* layer as data source will result in suboptimal performance). Our recommendation is using a 95% compression ratio for LMDB, or if you want to achieve maximum theoretical performance, don't use any data layer.

Change the batchsize in prototxt files. With some system configurations, higher batchsize will lead to better results.

The current implementation uses OpenMP threads. By default, the number of OpenMP threads is set to be equal to the number of CPU cores. Each one thread is bound to a single core to achieve best performance results. However, it is possible to use your own configuration by providing the right configuration through the OpenMP environmental variables like ``KMP_AFFINITY``, ``OMP_NUM_THREADS`` or ``GOMP_CPU_AFFINITY``. For single-node systems based on the Intel Xeon Phi™ product family, we recommend using ``OMP_NUM_THREADS = numer_of_cores-2``.

Additional topologies
---------------------

.. _Note:
   [I added this section as well as the General performance messages and Common issues sections in case we need additional description. We can remove/change/add these section according to your expertise]


General performance messages
----------------------------

Contact the Caffe team through the `team GitHub <https://github.com/intel/caffe/issues/>`_. 

Common issues
-------------

Contact the Caffe team through the `team GitHub <https://github.com/intel/caffe/issues/>`_. 

Training Examples
=================

.. _Note:
   [Please confirm this example is still relevent and complete. Point to models folder. Add sections on How to Train, How to do Inference, How to run benchmarks. Include CLI commands and execution results for examples in each section. Command below is to benchmark. Need similar sections for each framework. Also section for how to train on multinode. Caffe wiki has good documentation for this section. ]

::

  ./build/tools/caffe time --model=models/bvlc_googlenet/train_val.prototxt -iterations 100

To achieve results in images, follow the last section in the log, which is similar to the example below::

  Average Forward pass: xxx ms. Average Backward pass: xxx ms. Average Forward-Backward: xxx ms.

and use this equation::

  [Images/s] = batchsize * 1000 / Average Forward-Backward [ms]

Single-node Training
--------------------

Training and Resuming
^^^^^^^^^^^^^^^^^^^^^

While training the Intel Distribution of Caffe, two files that define the state of the network will be output:

  * .caffemodel
  * .solverstate

These two files define the current state of the network at a given iteration, and with this information we are able to continue training our network in the case of a hiccup, pause for diagnosis, or a system crash.

Training
^^^^^^^^

To begin training, we simply need to call the Caffe binary and supply a solver using the following command::

  caffe train -solver solver.prototxt
  Stopping

Number of Iterations Limit
^^^^^^^^^^^^^^^^^^^^^^^^^^

We can have our network stop after a specified number of iterations by providing a parameter in the *solver.prototxt* file named ``max_iter``.

For example, we can specify that we would like our network to stop after 60,000 iterations. We do this by setting the parameter accordingly::

   max_iter: 600000
  
Manually Stopping
^^^^^^^^^^^^^^^^^

It is possible to manually stop a network from training by pressing the Ctrl+C key combination. When the stop signal is sent, the network halts the forward and backwards pass and then outputs the current state of the network in *.caffemodel* and *.solverstate* files titled with the current iteration number.

Resuming
^^^^^^^^

When a network has stopped training, either due being manually halted or by reaching the set maximum iterations, we can resume training our network by telling Caffe to resume train from where we left off. This is as simple as supplying the snapshot flag with the current *.solverstate* file. For example::

  caffe train -solver solver.prototxt -snapshot train_190000.solverstate

In this case we will continue training from iteration 190,000.


Guide to multinode training with Intel® Distribution of Caffe*
---------------------------------------------------------------

This is an introduction to multi-node training with the Intel® Distribution of Caffe* framework. Supplementary information can be found in the GitHub wiki, and links to the wiki are provided in this guide. By the end of this guide, you should understand how multinode training is implemented in Intel® Distribution of Caffe* and be able to train any topology yourself on a simple cluster. Be sure to check out the performance optimization guidelines above.

To make the practical part of this guide more comprehensible, these instructions assume you have configured a cluster comprising four nodes from scratch. You will learn how to configure such a cluster, how to compile Intel® Distribution of Caffe*, how to run a training of a particular model, and how to verify the network actually has been trained.

How it works
^^^^^^^^^^^^

In case you are not interested in how multi-node in Intel® Distribution of Caffe* works and just want to run the training, please skip to the Configuring Cluster for Intel Distribution of Caffe section below.

Intel® Distribution of Caffe* is designed for both single node and multinode operation. We describe the multinode part here.

There are two general approaches to parallelization: Data parallelism and model parallelism. The approach used in Intel® Distribution of Caffe* is data parallelism.

Data parallelism
^^^^^^^^^^^^^^^^

Data parallelization runs training on different batches of data on each of the nodes. The data is split among all nodes but the same model is used. This means that the total batch size in a single iteration is equal to the sum of individual batch sizes of all nodes. For example, a network is trained on 8 nodes and all 8 nodes have a batch size of 128. The (total) batch size in a single iteration of the Stochastic Gradient Descent algorithm is 8*128=1024.

Intel® Distribution of Caffe* with MLSL offers two approaches for multinode training:

  * **Default**: Caffe does an ``Allreduce`` operation for gradients, and then each node does ``SGD`` locally, followed by ``Allgather`` for weights increments.
  * **Distributed weights update**: Caffe does a ``Reduce-Scatter`` operation for gradients, and then each node does ``SGD`` locally, followed by ``Allgather`` for weights increments.

Distribution of data
^^^^^^^^^^^^^^^^^^^^

One way to distribute data in a multinode cluster is to do the following:

  1. Divide your training data set into disjoint subsets of roughly equal size. 
  2. Distribute each subset into each node used for training. 
  3. Run the multinode training with the data layer prepared accordingly, which means either preparing separate proto configurations or placing each subset in exactly the same path for each node.

An easier approach is to simply distribute the full data set on all nodes and configure the data layer to draw different subsets on each node. Remember to set ``shuffle:true`` for the training phase in prototxt. Since each node has its own unique randomizing seed, it will effectively draw a unique image subset.

Communication
^^^^^^^^^^^^^

Intel® Distribution of Caffe* utilizes the Intel® Machine Learning Scaling Library (MLSL), which provides communication primitives for data parallelism and model parallelism, communication patterns for SGD and its variants (AdaGrad, Momentum, etc), and distributed weight updates. It is optimized for Intel® Xeon® and Intel® Xeon Phi (TM) processors and supports Intel® Omni-Path Architecture, Infiniband, and Ethernet. Refer to the `MLSL Wiki <https://github.com/01org/MLSL/>`_ or the `MLSL Developer Guide and Reference <https://github.com/01org/MLSL/blob/master/doc/Developer_Guide.pdf/>`_ for more details on the library.

Snapshots
^^^^^^^^^

Snapshots are saved only by the node hosting the root process (rank number 0). To resume training from a snapshot, the file has to be populated across all nodes participating in a training.

Test phase during training
^^^^^^^^^^^^^^^^^^^^^^^^^^

If *test phase* is enabled in the solver’s *protobuf* file, all the nodes are carrying out the tests and the results are aggregated by the ``Allreduce`` operation. The validation set needs to be present on every machine that has *test phase* specified in the solver *protobuf* file. This is important to remember when you want to use the same solver file on all machines instead of working with multiple protobuf files.

Configuring Cluster for Intel® Distribution of Caffe*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section explains how to configure a cluster, and what components to install to build Intel® Distribution of Caffe* to start distributed training using Intel® Machine Learning Scaling Library.

Hardware and software configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This guide assumes you are working on a cluster with four machines with IP addresses in the range from 192.161.32.1 to 192.161.32.4, and that you are starting the cluster from fresh installations of CentOS 7.2 64-bit. You can download the OS image for free from the `official website <https://www.centos.org/download/>`_. The minimal ISO is enough. Install the OS on all four nodes, and run the following command on each node to upgrade packages to the latest versions::

  # yum upgrade

TIP: You can also execute ``yum -y upgrade`` to suppress the prompt that asks for confirmation of the operation (unattended upgrade).

Preparing the system
^^^^^^^^^^^^^^^^^^^^

Before installing Intel® Distribution of Caffe* you need to install prerequisites. Start by choosing the master machine (in our example, this is the machine with the 192.161.32.1 address).

On each machine install “Extra Packages for Enterprise Linux”::

  # yum install epel-release
  # yum clean all

On the master machine install "Development Tools" and ansible::

  # yum groupinstall "Development Tools"
  # yum install ansible

Configuring ansible and ssh
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure ansible's inventory on the master machine by adding ``ourmaster`` and ``ourcluster`` sections in the */etc/ansible/* hosts file and fill in slave IPs::

  [ourmaster]
  192.161.31.1
  [ourcluster]
  192.161.32.[2:4]

On each slave machine configure SSH authentication using the master machine’s public key so that you can log in with ssh without a password. Generate an RSA key on master machine with this command::

  $ ssh-keygen -t rsa

Then copy the public part of the key to slave machines::

  $ ssh-copy-id -i ~/.ssh/id_rsa.pub 192.161.32.2
  $ ssh-copy-id -i ~/.ssh/id_rsa.pub 192.161.32.3
  $ ssh-copy-id -i ~/.ssh/id_rsa.pub 192.161.32.4

Verify ansible works by running the ping command from the master machine. The slave machines should respond with output similar to that shown below.::

  $ ansible ourcluster -m ping

Example output::

  192.168.31.2 | SUCCESS => {
      “changed“: false,
      “ping“: “pong“
  }
  192.168.31.3 | SUCCESS => {
      “changed“: false,
      “ping“: “pong“
  }
  192.168.31.4 | SUCCESS => {
      “changed“: false,
      “ping“: “pong“
  }

The master machine can also ping itself with the command ``ansible ourmaster -m ping``, and can ping the entire inventory by ``ansible all -m ping``.

Installing tools
^^^^^^^^^^^^^^^^

On the master machine, use ansible to install the packages listed by running the command below for the entire cluster.

::

  # ansible all -m shell -a 'yum -y install python-devel boost boost-devel cmake numpy \
  numpy-devel gflags gflags-devel glog glog-devel protobuf protobuf-devel hdf5 \
  hdf5-devel lmdb lmdb-devel leveldb leveldb-devel snappy-devel opencv opencv-devel'

Optionally you can install additional system tools that you might find useful.

::

  # ansible all -m shell -a 'yum install -y mc cpuinfo htop tmux screen iftop iperf \
  vim wget'

You might be required to turn off the firewall for each node (refer to `Firewalls and MPI <https://software.intel.com/en-us/articles/firewalls-and-mpi/>`_ for more information).

::

  # ansible all -m shell -a 'systemctl stop firewalld.service'

The cluster is ready to deploy binaries of Intel® Distribution of Caffe*. Let’s build it now.


Building Intel® Distribution of Caffe* on multinode systems
-----------------------------------------------------------

This section explains how to build the Intel® Distribution of Caffe* for multinode (distributed) training of neural networks. First, you need to install Intel® Machine Learning Scaling Library (Intel® MLSL).

Download the MLSL 2017 Update 1 Preview release package to the master machine. Use ansible to populate the installation package to the remaining cluster nodes.

::

  # ansible ourcluster -m synchronize -a \
  'src=~/intel-mlsl-devel-64-2017.1-016.x86_64.rpm dest=~/'

Install Intel MLSL on each node in the cluster using the following command:

::

  # ansible all -m shell -a 'rpm -i ~/intel-mlsl-devel-64-2017.1-016.x86_64.rpm'

Getting Intel® Distribution of Caffe* Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the master machine, execute the following command to obtain the latest snapshot of Intel® Distribution of Caffe*, including multinode support for distributed training.

::

  $ git clone https://github.com/intel/caffe.git intelcaffe

Preparing Environment before the Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure Intel® Machine Learning Scaling Library for the build.

::

  $ source /opt/intel/mlsl_2017.1.016/intel64/bin/mlslvars.sh

.. _Note:
   Building the Intel® Distribution of Caffe* will trigger Intel® Math Kernel Library for Machine Learning (MKLML) to be downloaded to the *intelcaffe/external/mkl/* directory and automatically configured.

Building from Makefile
----------------------

This section covers only the portion required to build Intel® Distribution of Caffe* with multinode support using Makefile. Refer to the `Caffe documentation <http://caffe.berkeleyvision.org/installation.html/>`_ for general information on how to build Caffe using Makefile.

Start by changing your working directory to the location where Intel® Distribution of Caffe* repository have been downloaded (for example, *~/intelcaffe*).

::

  $ cd ~/intelcaffe

Make a copy of *Makefile.config.example*, and rename it to *Makefile.config**::

  $ cp Makefile.config.example Makefile.config

Open *Makefile.config* in your favorite editor and uncomment the ``USE_MLSL`` variable.

::

  # Intel(r) Machine Learning Scaling Library (uncomment to build with MLSL)
  USE_MLSL := 1

Execute the ``make`` command to build Intel® Distribution of Caffe* with multinode support.

:: 

  $ make -j <number_of_physical_cores> -k

Building from CMake
-------------------

This section covers only the portion required to build Intel® Distribution of Caffe* with multinode support using CMake. Refer to the `Caffe documentation <http://caffe.berkeleyvision.org/installation.html/>`_ for general information on how to build Caffe using CMake. 

Start by changing work directory to the location where Intel® Distribution of Caffe* repository have been downloaded (for example, *~/intelcaffe*).

::

  $ cd ~/intelcaffe

Create the build directory and change work directory to build directory.

::

  $ mkdir build
  $ cd build

Execute the following CMake command to prepare the build.

::

  $ cmake .. -DBLAS=mkl -DUSE_MLSL=1 -DCPU_ONLY=1

Execute the make command to build Intel® Distribution of Caffe* with multi-node support.

::

  $ make -j <number_of_physical_cores> -k
  $ cd ..

Populating Caffe Binaries across Cluster Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After a successful build, synchronize the *intelcaffe* directories on the slave machines.

::

  $ ansible ourcluster -m synchronize -a ‘src=~/intelcaffe dest=~/’

Running Multinode Training with Intel® Distribution of Caffe*
-------------------------------------------------------------

Instructions on how to train CIFAR10 and GoogLeNet are explained in more details in the `Multi-node CIFAR10 tutorial <https://github.com/intel/caffe/wiki/multinode-cifar10/>`_ and `Multinode GoogleNet tutorial <https://github.com/intel/caffe/wiki/Multinode-googlenet/>`_. We recommend completing the CIFAR10 tutorial before you proceed with the instructions here. Here, GoogleNet will be trained on a four node cluster. If you want to learn more about GoogleNet training refer to the tutorials above.

Before you can train anything, you need to prepare the dataset. In this section, we assume that you have already downloaded the ImageNet training and validation datasets, and that they are stored on each node in the */home/data/imagenet/train* directory for the training set and */home/data/imagenet/val* for the validation set. For details, you can look at the Data Preparation section of BVLC Caffe examples at http://caffe.berkeleyvision.org/gathered/examples/imagenet.html. You can use your own data sets as well.

Next step is to create a machine file named *~/mpd.hosts* on the master node to control the placement of MPI process across the machines::

  192.161.32.1
  192.161.32.2
  192.161.32.3
  192.161.32.4

Update your model file *models/bvlc_googlenet/train_val_client.prototxt* as shown below::

 name: "GoogleNet"
 layer {
   name: "data"
   type: "ImageData"
   top: "data"
   top: "label"
   include {
   phase: TRAIN
   }
   transform_param {
   mirror: true
   crop_size: 224
   mean_value: 104
   mean_value: 117
   mean_value: 123
   }
   image_data_param {
   source: "/home/data/train.txt"
   batch_size: 256
   shuffle: true
   }
 }
 layer {
 name: "data"
 type: "ImageData"
 top: "data"
 top: "label"
 include {
 phase: TEST
   }
   transform_param {
   crop_size: 224
   mean_value: 104
   mean_value: 117
   mean_value: 123
   }
   image_data_param {
   source: "/home/data/val.txt"
   batch_size: 50
   new_width: 256
   new_height: 256
   }
 }

Synchronize the *intelcaffe* directories in all nodes, change your working directory to *intelcaffe*, and start the training process with the following command::

  $ mpirun -n 4 -ppn 1 -machinefile ~/mpd.hosts ./build/tools/caffe train \
  --solver=models/bvlc_googlenet/solver_client.prototxt --engine=MKL2017 2>&1 | tee -i ~/intelcaffe/multinode_train.out

The log from the training process will be written to the *multinode_train.out* file.

Test the trained network
------------------------

When the training is finished, you can test how your network has trained with the following command::

  $ ./build/tools/caffe test --model=models/bvlc_googlenet/train_val_client.prototxt 
  --weights=multinode_googlenet_iter_100000.caffemodel --iterations=1000

Look at the bottom lines of the output from the above command that contain ``loss3/top-1`` and ``loss3/top-5``. The values should be around ``loss3/top-1 = 0.69`` and ``loss3/top-5 = 0.886``.

For more information about Caffe testing, visit the Caffe interfaces website at http://caffe.berkeleyvision.org/tutorial/interfaces.html.


Testing and Inference
=====================

Testing also known as inference, classification, or scoring. Inference can be done in Python or by using the native C++ utility that ships with Caffe. To classify an image (or signal) or set of images the following information is needed::

    * Image(s)
    * Network architecture
    * Network weights

Testing using the native C++ utility is less flexible, so we prefer to use Python. The protoxt file with the model should have ``phase: TEST`` in the data layer with the testing dataset so we can test the model.

::

 /path/to/caffe/build/tools/caffe test -model /path/to/train_val.prototxt 
 - weights /path/to/trained_model.caffemodel -iterations <num_iter>

This example was adapted from this blog. To classify an image using a pretrained model, first download the pretrained model:

::

  ./scripts/download_model_binary.py models/bvlc_reference_caffenet

Next, download the dataset (*ILSVRC 2012* in this example) and labels (also called the *synset* file) which are required to map a prediction to the name of the class:

::

  ./data/ilsvrc12/get_ilsvrc_aux.sh

Then classify an image:

::

   ./build/examples/cpp_classification/classification.bin 
   models/bvlc_reference_caffenet/deploy.prototxt 
   models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel 
   data/ilsvrc12/imagenet_mean.binaryproto 
   data/ilsvrc12/synset_words.txt 
   examples/images/cat.jpg

The output should look like this:

::

  ---------- Prediction for examples/images/cat.jpg ----------
  0.3134 - "n02123045 tabby, tabby cat"
  0.2380 - "n02123159 tiger cat"
  0.1235 - "n02124075 Egyptian cat"
  0.1003 - "n02119022 red fox, Vulpes vulpes"
  0.0715 - "n02127052 lynx, catamount"


Benchmarks
==========

The Caffe Time 2.0 tool allows you to measure various performance indicators.

To use it you need to compile it with the ``–D PERFORMANCE_MONITORING=1`` flag. For example, run the following command::
 
  rm -fr build && mkdir build && cd build && export MKLDNNROOT="" && export MKLROOT=/opt/mklml_lnx_2017.0.2.20170110/ && cmake .. -DCPU_ONLY=ON -DUSE_MKL2017_AS_DEFAULT_ENGINE=ON -DPERFORMANCE_MONITORING=ON && make all -j && cd ..

Caffe has its own benchmark tool built in called Caffe Time. You can run it using the command below::

  ./build/tools/caffe time -model=models/default_googlenet_v2/train_val.prototxt

We created our own tool (Caffe Time 2.0) to make more precise measurements. To enable more thorough benchmarking you need to compile Caffe with the ``PERFORMANCE_MONITORING`` flag set and then run training. For example::

  ./build/tools/caffe train -solver=models/default_googlenet_v2/solver.prototxt

After training, output from our performance monitor will appear at the end of the output. The performance monitor output provides information about how much time (in nanoseconds) was spend on operations in each layer. It returns the average time, minimum time, and maximum time. There are two kinds of columns with suffixes *total* and *proc*. Data in *proc* columns show how much time was spend on calculations. Data in *total* also includes time for writing/reading, lags, etc

If you want to check how this is done in the code, take a look at the *performance.hpp header* file in *caffe/include/caffe/util/performance.hpp*. The most important parts are the functions defined at the top:

  * PERFORMANCE_CREATE_MONITOR
  * PERFORMANCE_INIT_MONITOR
  * PERFORMANCE_MEASUREMENT_BEGIN
  * PERFORMANCE_MEASUREMENT_END_STATIC

Static function is a performance tweak so that we decrease calls to ``getEventIdByName`` where we know that name won't change. (For example, ``mkl_conversion``) Also notice ``class Measurement``, which is implemented as a sort of stack. For this, some measurements are nested in other measurements, like in Intel MKL layers. For example, in *src/caffe/mkl_memory.cpp* you can see this in line 198 (call to ``PERFORMANCE_MEASUREMENT_BEGIN``) and line 200 (call to ``PERFORMANCE_MEASUREMENT_END_STATIC``)



Additional resources
====================

.. Note::
   [This section is for any additional links/information that users might find relevent/helpful. The more information we can provide, the better.]

https://github.com/intel/caffe/tree/master/models/intel_optimized_models