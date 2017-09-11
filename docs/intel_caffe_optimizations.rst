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

Release notes
=============

* 0.1 - Initial draft of Caffe performance optimization for Intel Architecture
* 0.2 - 

Introduction
============

.. Note::
   [In this section, we want to briefly describe the purpose, goals, and audience for this article.]

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (`BVLC <http://bvlc.eecs.berkeley.edu/>`_) and by community contributors. `Yangqing Jia <http://daggerfs.com/>`_ created the project during his PhD at UC Berkeley. Caffe is released under the `BSD 2-Clause license <https://github.com/BVLC/caffe/blob/master/LICENSE/>`_.

These instructions walk throught the configuration steps needed to optimize the Intel® distribution of Caffe* on platforms based on Intel® Xeon® processors code named Skylake. 

[[Need an actual processor number here: Skylake processors are: processor family E3-12xx v5 and E3-12xx v5 processors, Xeon E3-12xx v5  Xeon W-21xx Xeon Bronze, Silver, Gold, Platinum]]

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

We highly recommend using the Homebrew package manager. Ideally you could start from a clean /usr/local to avoid conflicts. In the following, we assume that you're using Anaconda Python and Homebrew.

CUDA
^^^^

Install via the NVIDIA package that includes both CUDA and the bundled driver. CUDA 7 is strongly suggested. Older CUDA require libstdc++ while clang++ is the default compiler and libc++ the default standard library on OS X 10.9+. This disagreement makes it necessary to change the compilation settings for each of the dependencies. This is prone to error.

Library Path
^^^^^^^^^^^^

We find that everything compiles successfully if $LD_LIBRARY_PATH is not set at all, and $DYLD_FALLBACK_LIBRARY_PATH is set to provide CUDA, Python, and other relevant libraries (e.g. /usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib). In other ENV settings, things may not work as expected.

General dependencies
~~~~~~~~~~~~~~~~~~~~

::

  brew install -vd snappy leveldb gflags glog szip lmdb
  # need the homebrew science source for OpenCV and hdf5
  brew tap homebrew/science
  brew install hdf5 opencv

If using Anaconda Python, a modification to the OpenCV formula might be needed. 

Do brew edit opencv and change the lines that look like the two lines below to exactly the two lines below::

  -DPYTHON_LIBRARY=#{py_prefix}/lib/libpython2.7.dylib
  -DPYTHON_INCLUDE_DIR=#{py_prefix}/include/python2.7

If using Anaconda Python, HDF5 is bundled and the hdf5 formula can be skipped.

Remaining dependencies, with / without Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  # with Python pycaffe needs dependencies built from source
  brew install --build-from-source --with-python -vd protobuf
  brew install --build-from-source -vd boost boost-python
  # without Python the usual installation suffices
  brew install protobuf boost

**BLAS**: already installed as the Accelerate / vecLib Framework. OpenBLAS and MKL are alternatives for faster CPU computation.

**Python** (optional): Anaconda is the preferred Python. If you decide against it, please use Homebrew. Check that Caffe and dependencies are linking against the same, desired Python.

Continue with compilation.


libstdc++ installation
^^^^^^^^^^^^^^^^^^^^^^

This route is not for the faint of heart. For OS X 10.10 and 10.9 you should install CUDA 7 and follow the instructions above. If that is not an option, take a deep breath and carry on.

In OS X 10.9+, clang++ is the default C++ compiler and uses libc++ as the standard library. However, NVIDIA CUDA (even version 6.0) currently links only with libstdc++. This makes it necessary to change the compilation settings for each of the dependencies.

We do this by modifying the Homebrew formulae before installing any packages. Make sure that Homebrew doesn't install any software dependencies in the background; all packages must be linked to libstdc++.

The prerequisite Homebrew formulae are::

  boost snappy leveldb protobuf gflags glog szip lmdb homebrew/science/opencv

For each of these formulas, brew edit FORMULA, and add the ENV definitions as shown::

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

If this is not done exactly right then linking errors will trouble you.

Homebrew versioning that Homebrew maintains itself as a separate git repository and making the above brew edit FORMULA changes will change files in your local copy of homebrew's master branch. By default, this will prevent you from updating Homebrew using brew update, as you will get an error message like the following::

  $ brew update
  error: Your local changes to the following files would be overwritten by merge:
    Library/Formula/lmdb.rb
  Please, commit your changes or stash them before you can merge.
  Aborting
  Error: Failure while executing: git pull -q origin refs/heads/master:refs/remotes/origin/master

One solution is to commit your changes to a separate Homebrew branch, run brew update, and rebase your changes onto the updated master. You'll have to do this both for the main Homebrew repository in /usr/local/ and the Homebrew science repository that contains OpenCV in /usr/local/Library/Taps/homebrew/homebrew-science, as follows::

  cd /usr/local
  git checkout -b caffe
  git add .
  git commit -m "Update Caffe dependencies to use libstdc++"
  cd /usr/local/Library/Taps/homebrew/homebrew-science
  git checkout -b caffe
  git add .
  git commit -m "Update Caffe dependencies"

Then, whenever you want to update homebrew, switch back to the master branches, do the update, rebase the caffe branches onto master and fix any conflicts::

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

**CUDA**: Install via the NVIDIA package instead of apt-get to be certain of the library and driver versions. Install the library and latest driver separately; the driver bundled with the library is usually out-of-date. This can be skipped for CPU-only installation.

**BLAS**: install ATLAS by sudo apt-get install libatlas-base-dev or install OpenBLAS or MKL for better CPU performance.

**Python** (optional): if you use the default Python you will need to sudo apt-get install the python-dev package to have the Python headers for building the pycaffe interface.

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

Note that glog does not compile with the most recent gflags version (2.1), so before that is resolved you will need to build with glog first.

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

Note that glog does not compile with the most recent gflags version (2.1), so before that is resolved you will need to build with glog first.

**CUDA**: Install via the NVIDIA package instead of yum to be certain of the library and driver versions. Install the library and latest driver separately; the driver bundled with the library is usually out-of-date. + CentOS/RHEL/Fedora:

**BLAS**: install ATLAS by sudo yum install atlas-devel or install OpenBLAS or MKL for better CPU performance. For the Makefile build, uncomment and set BLAS_LIB accordingly as ATLAS is usually installed under /usr/lib[64]/atlas).

**Python** (optional): if you use the default Python you will need to sudo yum install the python-devel package to have the Python headers for building the pycaffe wrapper.

Continue with compilation.


Installing Intel MKL-DNN
------------------------

[[This section follows the linked OS-dependent dependencies above. We might want to link out to the MKL-DNN and MKL installation instructions instead of duplicating them here.]]

.. _Note: Intel offers users the choice of using either `Intel MKL-DNN <https://github.com/intel/mklnn/>`_ for developers looking for an open source performance library for Deep Learning applications, or `Intel MKL <https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2017-install-guide/>`_ for developers who want a Intel-proprietary computing math library for applications that require maximum performance.

[[Installing MKL and MKL-DNN, can use either, open or closed source. Include prerequisites from existing documention, compiler, python librarires. MKL-DNN supports desktop and Atom processeors; do not have validaton ATM, but some optimizations are available. Our testing focuses on particular configuration, but MKL DNN does support other platforms as described in MKL DNN documentation. ]]

This section elaborates on the installation information presented on the GitHub repository site by providing detailed, step-by-step instructions for installing and building the Intel MKL-DNN library components. The computer you use will require an Intel® processor supporting Intel® Advanced Vector Extensions 2 (Intel® AVX2). Specifically, Intel MKL-DNN is optimized for Intel® Xeon® processors, Intel® Xeon Phi™ processors, and `Intel AVX-512 <https://www.intel.com/content/www/us/en/architecture-and-technology/avx-512-overview.html/>`_.

GitHub indicates the software was validated on RedHat* Enterprise Linux* 7; however, the information presented in this tutorial was developed on a system running Ubuntu* 16.04.

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

Intel MKL-DNN has the following dependencies:

    CMake* – a cross-platform tool used to build, test, and package software.
    Doxygen* – a tool for generating documentation from annotated source code.

If these software tools are not already set up on your computer you can install them with the following commands::

  sudo apt install cmake
  sudo apt install doxygen

Download and Build the Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the Intel MKL-DNN library from the GitHub repository by opening a terminal and typing the following command::

  git clone https://github.com/01org/mkl-dnn.git

Note: if Git* is not already set up on your computer you can install it by typing the following::

  sudo apt install git

Once the installation has completed you will find a directory named *mkl-dnn* in the Home directory. Navigate to this directory by typing::

  cd mkl-dnn

As explained on the GitHub repository site, Intel MKL-DNN uses the optimized general matrix to matrix multiplication (GEMM) function from Intel MKL. The library supporting this function is also included in the repository and can be downloaded by running the prepare_mkl.sh script located in the scripts directory::

  cd scripts && ./prepare_mkl.sh && cd ..

This script creates a directory named external and then downloads and extracts the library files to a directory named mkl-dnn/external/mklml_lnx*.

Execute the next command from the mkl-dnn directory. This command creates a subdirectory named *build* and then runs CMake and Make to generate the build system::

  mkdir -p build && cd build && cmake .. && make

Use the automated script to install additional libraries, list folders etc.
Other frameworks will require more description.

Include all relevent information here instead of in links. 


Building for Intel® Architecture
================================

[[Check with Caffe team to verify this procedure. Boost and GEM isn't optimized; works best with MKL and MKLDNN. Frank y zhang and Daisy Deng to find out how to buildng  MKL and MKL-DNN]]

.. Note::
   [This section should be after prerequisites. 
   Prerequisites
   Build
   Configure
   Run
   Optimize
   Examples
   ]

https://github.com/intel/caffe/blob/master/docs/release_notes.md#Building

This version of Caffe is optimized for Intel® Xeon processors and Intel® Xeon Phi™ processors. To achieve the best performance results on Intel Architecture we recommend building Intel® Distribution of Caffe* with Intel® MKL and enabling OpenMP support. This Caffe version is seflcontained. This means that newest version of Intel MKL will be downloaded and installed during compilation of Intel® Distribution of Caffe*.

Set ``BLAS := mkl`` in ``Makefile.config``

If you don't need GPU optimizations, ``CPU_ONLY := 1 flag`` in ``Makefile.config`` to configure and build Intel® Distribution of Caffe* without CUDA.

[Intel MKL 2017] introduces optimized Deep Neural Network (DNN) performance primitives that allow to accelerate the most popular image recognition topologies. Intel® Distribution of Caffe* can take advantage of these primitives and get significantly better performance results compared to the previous versions of Intel MKL. There are two ways to take advantage of the new primitives:

* Set layer engine to ``MKL2017`` in prototxt file (model). Only this specific layer will be accelerated with new primitives.
* Use ``-engine = MKL2017`` in command line as an option during execution of caffe (training, scoring, benchmark)


.. Note::
   {DO WE NEED TO INCLUDE INSTRUCTIONS ON BUILDING FOR GPU?}

Compilation
===========

Caffe can be compiled with either Make or CMake. Make is officially supported while CMake is supported by the community. Build procedure is the same as on bvlc-caffe-master branch. When OpenMP is available will be used automatically.

Compilation with Make
---------------------

Configure the build by copying and modifying the example Makefile.config for your setup. The defaults should work, but uncomment the relevant lines if using Anaconda Python.

::

  cp Makefile.config.example Makefile.config
  # Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)

::

  make all
  make test
  make runtest

For CPU & GPU accelerated Caffe, no changes are needed.
For cuDNN acceleration using NVIDIA's proprietary cuDNN software, uncomment the USE_CUDNN := 1 switch in Makefile.config. cuDNN is sometimes but not always faster than Caffe's GPU acceleration.
For CPU-only Caffe, uncomment CPU_ONLY := 1 in Makefile.config.

To compile the Python and MATLAB wrappers do make pycaffe and make matcaffe respectively. Be sure to set your MATLAB and Python paths in Makefile.config first!

**Distribution**: run make distribute to create a distribute directory with all the Caffe headers, compiled libraries, binaries, etc. needed for distribution to other machines.

**Speed**: for a faster build, compile in parallel by doing make all -j8 where 8 is the number of parallel threads for compilation (a good choice for the number of threads is the number of cores in your machine).

Now that you have installed Caffe, check out the MNIST tutorial and the reference ImageNet model tutorial.


Compilation with CMake
----------------------

In lieu of manually editing Makefile.config to configure the build, Caffe offers an unofficial CMake build thanks to @Nerei, @akosiorek, and other members of the community. It requires CMake version >= 2.8.7. The basic steps are as follows::

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

Builing the caffe with Intel Compiler allows you to take full advantage of the Intel(R) processor. This is a step-by-step tutorial for building Intel caffe with MKLDNN library.

1. Building Boost library.

    Download the boost from offical page and unzip it.
    Execute the following commands step by step.::

        Run source <compiler root>/bin/compilervars.sh {ia32 OR intel64} or source <compiler root>/bin/compilervars.csh {ia32 OR intel64}
        cd <boost root>
        ./bootstrap.sh
        ./b2 install --prefix=<Boost.Build install dir>
    
    For 32-bit::

        ./b2 --build-dir=<Boost object directory> toolset=intel stage

    For 64-bit::

        ./b2 --build-dir=<Boost object directory> address-model=64 toolset=intel stage

2. Update Caffe's code for Intel Compiler supporting.

    We need to add the -xHost flag to the compiler flag settings for better performance on Intel(R) processor:

        * Added the ``-xHost`` to the variable ``CXX_HARDENING_FLAGS`` on line 373 of ``/path/to/caffe/Makefile``.
        * Added the ``-xHost`` to the variable ``COMMON_FLAGS`` on line 428 of ``/path/to/caffe/Makefile``
        * Modified the line 46 and 53 to ``$(eval CXXFLAGS += -DMKLDNN_SUPPORTED -xHost)`` of ``/path/to/caffe/Makefile.mkldnn``

3. Build caffe:

    * cd to ``/path/to/caffe`` and create the Makefile.config from the Makefile.config.example.
    * Set the variable ``CUSTOM_CXX`` to ``/path/to/icpc.`` For example: ``CUSTOM_CXX := /opt/intel/compilers_and_libraries/linux/bin/intel64/icpc``
    * Set the variable ``BOOST_ROOT`` to ``/path/to/unzipped_boost_root.`` For example: ``BOOST_ROOT := /home/user/boost_1_64_0``
    * Run ``make all -j$(nproc)`` to build the caffe.


Configuration
=============

.. Note::
   [Please confirm the instructions below are correct and complete.]

To achieve the best performance with the Intel® distribution of Caffe* on Intel processors please apply the following recommendations:

Hardware / BIOS configuration
-----------------------------

* Make sure that your hardware configurations includes fast SSD (M.2) drive. If during  trainings/scoring you will observe in logs "waiting for data"
  * You should install better SSD or reduce batchsize.
* With Intel Xeon Phi™ product family:
    * Enter BIOS (MCDRAM section) and set MCDRAM mode as cache
    *  Enable Hyper-treading (HT) on your platform - those setting can be found in BIOS (CPU  section).
* Optimize hardware in BIOS: 
    * set CPU max frequency
    * set 100% fan speed
    * check cooling system
* For multinode Intel Xeon Phi™ product family over Intel® Omni-Path Architecture use:
    * Processor C6 = Enabled
    * Snoop Holdoff Count = 9
    * Intel Turbo Boost Technology = Enabled
    * Uncore settings: Cluster Mode: All2All

Software / OS configuration
---------------------------

With Intel Xeon Phi™ product family:

* It is recommended to use Linux Centos 7.2 or newer.
* It is recommended to use newest XPPSL software for Intel Xeon Phi™ product family. 
    * https://software.intel.com/en-us/articles/xeon-phi-software#downloads
    * https://software.intel.com/en-us/articles/xeon-phi-software#downloads

* For multinode Intel Xeon Phi™ product family over Intel® Omni-Path Architecture use:
    * irqbalance needs to be installed and configured with --hintpolicy=exact option
    * CPU frequency needs to be set via intel_pstate driver::

          echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct
          echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
          cpupower frequency-set -g performance```

Make sure that there are no unnecessary processes during training and scoring. Intel® Distribution of Caffe* is using all available resources and other processes (like monitoring tools, java processes, network traffic etc.) might impact performance.

We recommend to compile Intel® Distribution of Caffe* with gcc 4.8.5 (or newer)

We recommend to compile Intel® Distribution of Caffe* with makefile.configuration set to::

    CPU_ONLY := 1
    BLAS := mkl

Intel® Distribution of Caffe / Hyper-Parameters configuration*
--------------------------------------------------------------

[These are examples. Say that you can find examples in models/IA-optimized-models folder, don't duplicate that information here. Links here, remove text from this section]

We provide two sets of prototxt files with Hyper-Parameters and network topologies. In default set you will find standard topologies and their configuration used by community. In BKM (Best Know Method) you will find our internaly developed solution optimized for Intel MKL2017 and Intel CPU.

When running performance and trainings - we recommend to start working with default sets to establish baseline.

Use LMDB data layer (Using ‘Images’ layer as data source will result in suboptimal performance). Our recommendation is to use 95% compression ratio for LMDB, or to achieve maximum theoretical performance - don't use any data layer.

Change batchsize in prototxt files. On some configurations higher batchsize will leads to better results.

Current implementation uses OpenMP threads. By default the number of OpenMP threads is set to the number of CPU cores. Each one thread is bound to a single core to achieve best performance results. It is however possible to use own configuration by providing right one through OpenMP environmental variables like KMP_AFFINITY, OMP_NUM_THREADS or GOMP_CPU_AFFINITY. For Intel Xeon Phi™ product family single-node we recommend to use OMP_NUM_THREADS = numer_of_corres-2.

[remove]
Our Recommended rules for setting Hyper Parameters for googlenet v1::

    Batch_per_node <=128

Learning Rate (LR) for total_batch: ``LR=0.07-0.08 for 1024 / LR=0.03 for 256 / LR=0.005 for 32`` (and you can rescale it for any total batch).

``Number_of_iterations * number_of_nodes * batch_per_node = 32 * 2.400.000`` (If you wish to achieve slightly better accuracy use ``32 * 2.400.000 * 1.2``)

Our multinode configuration for 8 nodes::

    batch_size_per_node = 128
    base_lr: 0.07
    max_iter: 75000
    lr_policy: "poly"
    power: 0.5
    momentum: 0.9
    weight_decay: 0.0002
    solver_mode: CPU

It is possible to speed-up training by Convolution weights initialization with Gabor Filters

Intel distribution of Caffe Benchmark Intel distribution of Caffe allow user to easly benchmark any topology and check their perofmrance. To run it just enter the command::

    caffe time --model=[path_path_to_prtotxt_file_with_model] -iterations [number_of_iterations]

,/remove]

Additional topologies
---------------------

.. Note::
   [I added this section as well as the General performance messages and Common issues sections in case we need additional description. We can remove/change/add these section according to your expertise]


General performance messages
----------------------------

Contact the Caffe team through the `team GitHub <https://github.com/intel/caffe/issues/>`_. 

[MORE INFO?]

Common issues
-------------

Contact the Caffe team through the `team GitHub <https://github.com/intel/caffe/issues/>`_. 

[MORE INFO?]



Example
=======

.. Note::
   [Please confirm this example is still relevent and complete. Point to models folder. Add sections on How to Train, How to do Inference, How to run benchmarks. Include CLI commands and execution results for examples in each section. Command below is to benchmark. Need similar sections for each framework. Also section for how to train on multinode. Caffe wiki has good documentation for this section. ]

::

  ./build/tools/caffe time --model=models/bvlc_googlenet/train_val.prototxt -iterations 100

To achieve results in images/s follow find last section in the log:

Average Forward pass: xxx ms. Average Backward pass: xxx ms. Average Forward-Backward: xxx ms.

and use equation::

  [Images/s] = batchsize * 1000 / Average Forward-Backward [ms]

Training
--------

Training and Resuming
^^^^^^^^^^^^^^^^^^^^^

In caffe, during training files defining the state of the network will be output: .caffemodel and .solverstate. These two files define the current state of the network at a given iteration, and with this information we are able to continue training our network in the case of a hiccup, pause for diagnosis, or a system crash.
Training

To begin training, we simply need to call the caffe binary and supply a solver::

  caffe train -solver solver.prototxt
  Stopping

Number of Iterations Limit
^^^^^^^^^^^^^^^^^^^^^^^^^^

We can have our network stop after a specified number of iterations with a parameter in the solver.prototxt named max_iter.

For example, we can specify that we would like our network to stop after 60,000 iteration, thus we set the parameter accordingly: max_iter: 600000.
Manually Stopping

It is possible to manually stop a network from training by pressing the Ctrl+C key combination. When the stop signal is sent, the network will halt the forward and backwards pass, and output the current state of the network in a .caffemodel and .solverstate titled with the current iteration number.
Resuming

When a network as stopped training, either due to manual halting or by reaching the maximum iterations, we may continue training our network by telling caffe to train from where we left off. This is as simple as supplying the snapshot flag with the current .solverstate file. For example::

  caffe train -solver solver.prototxt -snapshot train_190000.solverstate

In this case we will continue training from iteration 190000.


Guide to multi-node training with Intel® Distribution of Caffe*
---------------------------------------------------------------

This is an introduction to multi-node training with Intel® Distribution of Caffe* framework. All other pages related to multi-node in this wiki are supplementary and they are referred to in this guide. By the end of it, you should understand how multi-node training was implemented in Intel® Distribution of Caffe* and be able to train any topology yourself on a simple cluster. Basic knowledge of BVLC Caffe usage might be necessary to understand it fully. Also be sure to check out the performance optimization guidelines.
Introduction

To make the practical part of this guide more comprehensible, the instructions assume you have configured from scratch a cluster comprising 4 nodes. You will learn how to configure such a cluster, how to compile Intel® Distribution of Caffe*, how to run a training of a particular model, and how to verify the network actually have trained.
How it works

In case you are not interested in how multi-node in Intel® Distribution of Caffe* works and just want to run the training, please skip to the practical part chapter of this Wiki.

Intel® Distribution of Caffe* is designed for both single-node and multi-node operation. Here, the multi-node part is explained.

There are two general approaches to parallelization. Data parallelism and model parallelism. The approach used in Intel® Distribution of Caffe* is the data parallelism.

Data parallelism
^^^^^^^^^^^^^^^^

The data parallelization technique runs training on different batches of data on each of the nodes. The data is split among all nodes but the same model is used. It means that the total batch size in a single iteration is equal to the sum of individual batch sizes of all nodes. For example a network is trained on 8 nodes. All of them have batch size of 128. The (total) batch size in a single iteration of the Stochastic Gradient Descent algorithm is 8*128=1024.

Intel® Distribution of Caffe* with MLSL offers two approaches for multi-node training:

    Default - Caffe does Allreduce operation for gradients and then each node is doing SGD locally, followed by Allgather for weights increments.
    Distributed weights update - Caffe does Reduce-Scatter operation for gradients, then each node is doing SGD locally, followed by Allgather for weights increments.

Distribution of data
^^^^^^^^^^^^^^^^^^^^

One approach is to divide your training data set into disjoint subsets of roughly equal size. Distribute each subset into each node used for training. Run the multinode training with data layer prepared accordingly, which means either preparing separate proto configurations or placing each subset in exactly the same path for each node.

An easier approach is to simply distribute the full data set on all nodes and configure data layer to draw different subset on each node. Remember to set ``shuffle:true`` for the training phase in prototxt. Since each node has its own unique randomizing seed, it will effectively draw unique image subset.
Communication

Intel® Distribution of Caffe* is utilizing Intel® Machine Learning Scaling Library (MLSL) which provides communication primitives for data parallelism and model parallelism, communication patterns for SGD and its variants (AdaGrad, Momentum, etc), distributed weight update. It is optimized for Intel® Xeon® and Intel® Xeon Phi (TM) processors and supports Intel® Omni-Path Architecture, Infiniband and Ethernet. Refer to MLSL Wiki or "MLSL Developer Guide and Reference" for more details on the library.

Snapshots
^^^^^^^^^

Snapshots are saved only by the node hosting the root process (rank number 0). In order to resume training from a snapshot the file has to be populated across all nodes participating in a training.

Test phase during training
^^^^^^^^^^^^^^^^^^^^^^^^^^

If test phase is enabled in the solver’s protobuf file all the nodes are carrying out the tests and results are aggregated by Allreduce operation. The validation set needs to be present on every machine which have test phase specified in solver protobuf file. This is important because when you want to use the same solver file on all machines instead of working with multiple protobuf files you need to remember about that.

Configuring Cluster for Intel® Distribution of Caffe*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This chapter explains how to configure a cluster, and what components to install in order to build Intel® Distribution of Caffe* to start distributed training using Intel® Machine Learning Scaling Library.

Hardware and software configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hardware assumptions for this guide: 4 machines with IP addresses in the range from 192.161.32.1 to 192.161.32.4 up the cluster Start from fresh installation of CentOS 7.2 64-bit. The OS image can be downloaded free of charge from the official website. Minimal ISO is enough. You should install the OS on each node (all 4 in our example). Next upgrade to the latest version of packages (do it on each node)::

  # yum upgrade

TIP: You can also execute yum -y upgrade to suppress the prompt asking for confirmation of the operation (unattended upgrade).
Preparing the system

Before installing Intel® Distribution of Caffe* you need to install prerequisites. Start by choosing the master machine (e.g. 192.161.32.1 in our example).

On each machine install “Extra Packages for Enterprise Linux”::

  # yum install epel-release
  # yum clean all

On master machine install "Development Tools" and ansible::

  # yum groupinstall "Development Tools"
  # yum install ansible

Configuring ansible and ssh
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure ansible's inventory on master machine by adding sections ourmaster and ourcluster in /etc/ansible/ hosts and fill in slave IPs::

  [ourmaster]
  192.161.31.1
  [ourcluster]
  192.161.32.[2:4]

On each slave machine configure SSH authentication using master machine’s public key, so that you can log in with ssh without a password. Generate RSA key on master machine::

  $ ssh-keygen -t rsa

And copy the public part of the key to slave machines::

  $ ssh-copy-id -i ~/.ssh/id_rsa.pub 192.161.32.2
  $ ssh-copy-id -i ~/.ssh/id_rsa.pub 192.161.32.3
  $ ssh-copy-id -i ~/.ssh/id_rsa.pub 192.161.32.4

Verify ansible works by running ping command from master machine. The slave machines should respond.::

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

Master machine can also ping itself by ansible ourmaster -m ping and entire inventory by ansible all -m ping.

Installing tools
^^^^^^^^^^^^^^^^

On master machine use ansible to install packages listed by running the command below for the entire cluster.

::

  # ansible all -m shell -a 'yum -y install python-devel boost boost-devel cmake numpy \
  numpy-devel gflags gflags-devel glog glog-devel protobuf protobuf-devel hdf5 \
  hdf5-devel lmdb lmdb-devel leveldb leveldb-devel snappy-devel opencv opencv-devel'

Optionally you can install additional system tools you may find useful.

::

  # ansible all -m shell -a 'yum install -y mc cpuinfo htop tmux screen iftop iperf \
  vim wget'

You might be required to turn off the firewall on each node (refer to Firewalls and MPI for more information), too.

::

  # ansible all -m shell -a 'systemctl stop firewalld.service'

The cluster is ready to deploy binaries of Intel® Distribution of Caffe*. Let’s build it now.


Building Intel® Distribution of Caffe*
--------------------------------------

This chapter explains how to build Intel® Distribution of Caffe* for multi-node (distributed) training of neural networks.
Installing Intel® Machine Learning Scaling Library

Download the MLSL 2017 Update 1 Preview release package to master machine. Use ansible to populate installation package to the remaining cluster nodes.

::

  # ansible ourcluster -m synchronize -a \
  'src=~/intel-mlsl-devel-64-2017.1-016.x86_64.rpm dest=~/'

Install MLSL on each node in the cluster.

::

  # ansible all -m shell -a 'rpm -i ~/intel-mlsl-devel-64-2017.1-016.x86_64.rpm'

Getting Intel® Distribution of Caffe* Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On master machine execute the following git command in order to obtain the latest snapshot of Intel® Distribution of Caffe* including multi-node support for distributed training.

::

  $ git clone https://github.com/intel/caffe.git intelcaffe

Preparing Environment before the Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure Intel® Machine Learning Scaling Library for the build.

::

  $ source /opt/intel/mlsl_2017.1.016/intel64/bin/mlslvars.sh

.. Note::
   Build of Intel® Distribution of Caffe* will trigger Intel® Math Kernel Library for Machine Learning (MKLML) to be downloaded to the *intelcaffe/external/mkl/* directory and automatically configured.

Building from Makefile
----------------------

This section covers only the portion required to build Intel® Distribution of Caffe* with multi-node support using Makefile. Please refer to Caffe documentation for general information on how to build Caffe using Makefile.

Start by changing work directory to the location where Intel® Distribution of Caffe* repository have been downloaded (e.g. ~/intelcaffe).

::

  $ cd ~/intelcaffe

Make a copy of Makefile.config.example, and name it ``Makefile.config``::

  $ cp Makefile.config.example Makefile.config

Open Makefile.config in your favorite editor and uncomment the ``USE_MLSL`` variable.

::

  # Intel(r) Machine Learning Scaling Library (uncomment to build with MLSL)
  USE_MLSL := 1

Execute make command to build Intel® Distribution of Caffe* with multi-node support.

:: 

  $ make -j <number_of_physical_cores> -k

Building from CMake
-------------------

This section covers only the portion required to build Intel® Distribution of Caffe* with multi-node support using CMake. Please refer to Caffe documentation for general information on how to build Caffe using CMake. Start by changing work directory to the location where Intel® Distribution of Caffe* repository have been downloaded (e.g. ~/intelcaffe).

::

  $ cd ~/intelcaffe

Create build directory and change work directory to build directory.

::

  $ mkdir build
  $ cd build

Execute the following CMake command in order to prepare the build

::

  $ cmake .. -DBLAS=mkl -DUSE_MLSL=1 -DCPU_ONLY=1

Execute make command to build Intel® Distribution of Caffe* with multi-node support.

::

  $ make -j <number_of_physical_cores> -k
  $ cd ..

Populating Caffe Binaries across Cluster Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After successful build synchronize intelcaffe directories on the slave machines.

::

  $ ansible ourcluster -m synchronize -a ‘src=~/intelcaffe dest=~/’

Running Multi-node Training with Intel® Distribution of Caffe*
==============================================================

Instructions on how to train CIFAR10 and GoogLeNet are explained in more details in Multi-node CIFAR10 tutorial and Multi-node GoogLeNet tutorial. It is recommended to do CIFAR10 tutorial before you proceed. Here, the GoogLeNet will be trained on 4 node cluster. If you want to learn more about GoogLeNet training see the tutorial mentioned above as well.

Before you can train anything you need to prepare the dataset. It is assumed that you have already downloaded the ImageNet training and validation datasets, and they are stored on each node in /home/data/imagenet/train directory for training set and /home/data/imagenet/val for validation set. For details you can look at the Data Preparation section of BVLC Caffe examples at http://caffe.berkeleyvision.org/gathered/examples/imagenet.html. You can use your own data sets as well.

Next step is to create machine file ~/mpd.hosts on master node for controlling the placement of MPI process across the machines::

  192.161.32.1
  192.161.32.2
  192.161.32.3
  192.161.32.4

Update your model file models/bvlc_googlenet/train_val_client.prototxt::

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

Synchronize the intelcaffe directories, change your working directory to intelcaffe and start the training process with the following command::

  $ mpirun -n 4 -ppn 1 -machinefile ~/mpd.hosts ./build/tools/caffe train \
  --solver=models/bvlc_googlenet/solver_client.prototxt --engine=MKL2017 2>&1 | tee -i ~/intelcaffe/multinode_train.out

Log from the training process will be written to multinode_train.out file.

Test the trained network
------------------------

When the training is finished, you can test how your network has trained with the following command::

  $ ./build/tools/caffe test --model=models/bvlc_googlenet/train_val_client.prototxt 
  --weights=multinode_googlenet_iter_100000.caffemodel --iterations=1000

Look at the bottom lines of output from the above command which contains loss3/top-1 and loss3/top-5. The values should be around ``loss3/top-1 = 0.69`` and ``loss3/top-5 = 0.886``.

For more information about caffe test visit Caffe interfaces website at http://caffe.berkeleyvision.org/tutorial/interfaces.html.


Testing and Inference
---------------------

Testing also known as inference, classification, or scoring can be done in Python or using the native C++ utility that ships with Caffe. To classify an image (or signal) or set of images the following is needed::

    Image(s)
    Network architecture
    Network weights

Testing using the native C++ utility is less flexible, and using Python is preferred. The protoxt file with the model should have phase: TEST in the data layer with the testing dataset in order to test the model.

::

 /path/to/caffe/build/tools/caffe test -model /path/to/train_val.prototxt 
 - weights /path/to/trained_model.caffemodel -iterations <num_iter>

This example was adapted from this blog. To classify an image using a pretrained model, first download the pretrained model:

::

  ./scripts/download_model_binary.py models/bvlc_reference_caffenet

Next, download the dataset (ILSVRC 2012 in this example) labels (also called the synset file) which is required in order to map a prediction to the name of the class:

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

To use it you need to compile with compile with the ``–D PERFORMANCE_MONITORING=1`` flag. For example::
 
  rm -fr build && mkdir build && cd build && export MKLDNNROOT="" && export MKLROOT=/opt/mklml_lnx_2017.0.2.20170110/ && cmake .. -DCPU_ONLY=ON -DUSE_MKL2017_AS_DEFAULT_ENGINE=ON -DPERFORMANCE_MONITORING=ON && make all -j && cd ..

Caffe has benchmark tool built in, its called caffe time. You can run it for example using below command::

  ./build/tools/caffe time -model=models/default_googlenet_v2/train_val.prototxt

We created our own tool (caffe time 2.0) to make more precise measurements. To enable more thorough benchmark you need to compile caffe with the ``PERFORMANCE_MONITORING`` flag set and then run training. For example::

  ./build/tools/caffe train -solver=models/default_googlenet_v2/solver.prototxt

After training output from our performance monitor will appear at the end of the output. It provides info about how much time in nanoseconds was spend on operations in each layer. It returns average time, minimum, maximum. There are two kinds of columns with suffix total and proc. Data in proc columns show how much time was spend on calculations, total also includes time for writing/reading, lags etc

If you want to check how it is done in the code, take a look at performance.hpp header file in caffe/include/caffe/util/performance.hpp. The most important are functions defined at the top PERFORMANCE_CREATE_MONITOR, PERFORMANCE_INIT_MONITOR, PERFORMANCE_MEASUREMENT_BEGIN, PERFORMANCE_MEASUREMENT_END_STATIC. (Static function is a performance tweak so that we decrease calls to getEventIdByName where we know that name won't change. For example - mkl_conversion) Also notice class Measurement which is implemented as a sort of stack. It is for that there are some measurements nested in other measurements, ie. in MKL layers. For example in src/caffe/mkl_memory.cpp you can see in line 198 (call to PERFORMANCE_MEASUREMENT_BEGIN) and line 200 (call to PERFORMANCE_MEASUREMENT_END_STATIC)



Additional resources
====================

.. Note::
   [This section is for any additional links/information that users might find relevent/helpful. The more information we can provide, the better.]

https://github.com/intel/caffe/tree/master/models/intel_optimized_models