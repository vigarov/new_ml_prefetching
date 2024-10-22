# Setting up `fltrace`

This short guide will show you how to setup the `fltrace` tracing utility. This has been tested on Ubuntu 22.04.5 LTS running Linux 6.8.0-45. 

In the second part of this tutorial, a custom Linux kernel will be installed in order to get more data relevant data from userspace (e.g.: instruction pointer, registers). 

# Part I - getting the tool 

The tool's original Github repo is [here](https://github.com/eden-farmem/fltrace). However, some dependencies aren't correctly installed on it. You can therefore clone [our fork here](https://github.com/vigarov/fltrace). Make sure to use the `fixed_dependencies` branch (`git switch fixed_dependencies`)

Once you have downloaded the repository, you can make the tool by simple running 

```
$ make clean
$ make
```
# Part II - tracing the program

Say you want to trace an executable program `prog` located at `/path/to/prog`. To trace it, you can simply run (from inside the `fltrace` repository folder) 
```sh
sudo ./fltrace record -M <max_memory> -L <memory_limit> -- /path/to/prog
```
The `max_memory` ($M$) and `memory_limit` ($L$), both in MB, are two variables you have control of. 

* 
    $M$ should act as an upper limit to the program's maximum memory usage. Several great tools exist to get the memory usage of a program over time: two of note are `/usr/bin/time -v` (! different from the simpler `time` command) and `valgrind`'s [`massif`](https://valgrind.org/docs/manual/ms-manual.html) tool. In this project, we used valgrind as it provides slightly more consistent results (over 100 runs of the same application, it always computed the same value, whereas `/usr/bin/time` had the very small standard error of 0.00523). The difference betweend the tools was statistically insignificant however (p<0.0001), so feel free to use whichever you find easier!

    Before running `./fltrace`, you can therefore preliminarily trace your program using valgrind:

    ```
    valgrind --tool=massif /path/to/prog
    ```
    Note: take into account that running a program under valgrind greatly reduces it's execution speed. 

    This will create a `massif.out.*` file which you can analyse using the `ms_print massif.out.filename` utility. It will print a bunch of data, the graph we're interested in will look like this:

    ```

        MB
    256.0^:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
        |:                                                                      #
    0 +----------------------------------------------------------------------->Gi
        0                                                                   21.22
    ```
    The X-axis represents the instructions (it can be interpreted as time), whereas the Y-axis represents memory usage. For this particular example program, a simple linear array access was running, and we can therefore see that 256MB of memory was allocated from the start, and it was used (not deallocated) until the end of the program's runtime. 

    For this program, we would therefore pick $M \geq 256$, namely, $M=400$ to be safe. There are no known drawbacks of picking a too large $M$, however picking a too small $M$ will cause `fltrace` to crash (as its allocator will not be able to allocate enough memory upon request), so better safe than sorry!

    One other example graph can be found on [valgrind's website](https://valgrind.org/docs/manual/ms-manual.html).

*
    $L$ denotes the amount of locally available memory. From the program's point of view, this corresponds to the maximum amount of in-memory data. $L$ therefore acts as a memory limiter to your traced program (you can think of it as limiting your program's memory through `cgroups`). We must have $M \geq L$. Setting $L \coloneqq M$ means that your program will not be memory restricted when running.

    To get interesting results however, it is interesting to set $L < M$. The lower the value for $L$, the more page faults a program will encounter. Be careful to not set $L$ too low however, as some parts of your traced program might need a minimum amount of memory, and your program might segfault if it doesn't get them. Always monitor the output of your trace to ensure this is not the case!

    We have experimentally found $L = \frac{M}{4}$ to be a great compromise between getting a good amount of page faults (but not too much to not get too heavy traces) while ensuring the program never crashes.

Once `./fltrace` finishes to run, you will get 3 files as output in your working directory :

1. `fltrace-data-faults-<PID>.1.out`: contains the pagefault data in csv format
2. `fltrace-data-procmaps-<PID>.out`: contains a copy of the program' maps (`/proc/<PID>/maps`)
3. `fltrace-data-stats-<PID>.out`: contains several stats covering the program's execution (e.g.: binning of # faults, amount of memory freed, ...)

## (Bonus) Tracing PARSEC

PARSEC is a benchmark suite widely used in memory/OS research, introduced by [Princeton researchers in 2008](https://doi.org/10.1145/1454115.1454128). As of November 2023, the official Princeton website was down. The last working archive was from [September 22, 2023](https://web.archive.org/web/20230922200507/https://parsec.cs.princeton.edu/), and was unable to be built on modern devices (had several build errors). PARSEC can therefore be installed using [this maintained mirror](https://github.com/cirosantilli/parsec-benchmark).

After building the tool and setting up your environment, to run one benchmark, you should be able to execute (from anywhere) 
```sh
parsecmgmt -a run -i simulation_set_size -p benchmark_name
```

`parsecmgmt` is a handy shell script which has an additional useful command line argument, `-s`, which can be used to specify what program will be used to run the benchmark. By default, it is `/usr/bin/time`.

To `fltrace` a PARSEC benchmark, say `canneal` with a large simulation set, you can therefore run
```sh
parsecmgmt -a run -i simlarge -p canneal -s "sudo /path/to/fltrace record -M 200 -L 100 -- " 
```

In order to automatically gather data from many benchmarks, using different $M$,$L$ values, we have created a simple helper python script which executes fltrace and puts the output files in a relevant directory. This `execute_fltrace.py` can be found [here](helpers/execute_fltrace.py). Then, to trace a given set of benchmarks with all $(M,L)$ pairs s.t. $L=\frac{k}{4}M \ \  \forall k \in \{1,2,3,4\}$ by executing this monstrosity of a one-liner:

```sh
strings=("canneal" "ferret" "facesim" "bodytrack" "dedup" "fluidanimate" "raytrace" "streamcluster"); values=(200 150 500 30 1000 500 300 50); for ((i=0; i<${#strings[@]}; i++)); do string="${strings[$i]}"; N="${values[$i]}"; for ((j=0; j<4; j++)); do M=$((N * (j + 1) / 4)); parsecmgmt -a run -i simlarge -p "$string" -s "python /path/to/execute_fltrace.py --output_dir /home/user/data/raw/fltrace_out/\!BN\!/${N}_${M} --fltrace_path /path/to/fltrace $N $M "; done; done
```

# (Optional) Part III - extending the kernel to get more data

(i386 and x86_64 only)

By default, `userfaulfd` (which is the mechanism used by `fltrace` to capture page faults) does not include the instruction pointer of the program at fault time in its `uffd_msg` passed to the userfaultfd handler (see `struct uffd_msg`'s `arg.pagefault` in the kernel's /include/uapi/linux/userfaultfd.h definition). Executing the barebones `fltrace` version as above will therefore yield a value of `0` for the `ip` field (if you inspect the code, you will indeed notice that it isn't populated anywhere). 

Although this hasn't been documented anywhere for `fltrace`, we suspect the authors of the tool have used their [eden uffd-include-ip kernel patch](https://github.com/eden-farmem/eden/tree/master/kernel) to alleviate this. Since we want a more complete dataset, we extend this idea by including the whole `struct pt_regs`  - the registers from the fault frame obtained at page fault time - inside the `uffd_msg`. A guide to getting our custom kernel to built can be found [here](kernel.md).

Once the custom kernel has been installed, you can switch on the `reg_extending` branch of [our fltrace fork](https://github.com/vigarov/fltrace) :
```sh
git switch reg_extending
```
Then, simply remake the tool:
```sh
make clean
make
```

You will notice a build error: 
```
src/rmem/handler.c:10:10: fatal error: /usr/src/CUSTOM_KERNEL_NAME/include/uapi/linux/userfaultfd.h: No such file or directory
```

Simply replace `CUSTOM_KERNEL_NAME` in that source file by your custom's kernel name as listed under `/usr/src/` (e.g.: mine, following the guide, was `linux-hwe-6.8-headers-6.8.0-45+vgiuffd`). This manual change is needed (and can probably be automated through the Makefile, feel free to PR!), as when installing a custom kernel, the default headers (under `/usr/include/`) don't get replaced by the headers of your custom kernel.  

Now, tracing as before should yield a correctly populated value for the `ip` field under `fltrace-data-faults-<PID>.1.out` file, as well as the introduction of a new `regs` field, which contains the `struct pr_regs` register dump as it was populated by the fault handler. 

(ps: you can activate/deactive this new functionality through the `CFLAGS += -DEXTRA_REGS_KERNEL` line in the Makefile)