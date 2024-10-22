# Building your own kernel to extend userfaultfd

In order to get the instruction pointer and the rest of the trap frame's registers in `fltrace`, you must extend the kernel. To do so, you will need to build a custom kernel. 

This guide will provide step by step instructions on how to do so.

Note: it was created for machines (servers/PCs) running Ubuntu (22.04.5, but version shouldn't really matter). If your machine is running a different distro, the instruction will probably be different.

## I. Getting the source code and other prerequisites

You might be tempted to get the Linux source code from the [GitHub mirror](https://github.com/torvalds/linux), or the [official git source](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/). However, if you do so, getting the build to actually... build, and then run (signing if you have SecureBoot, building _all_ modules (*hum hum* nvidia *hum*)) will be a pain (and a potential huge loss of time).

I therefore recommend following [this guide](https://wiki.ubuntu.com/Kernel/BuildYourOwnKernel). If you do, an important note for the rest of this tutorial is to replace any occurence of `$(uname -r)` by the kernel version release you actually want (otherwise you will simply be getting the sources of the version currently running on your machine).

1. First, you will need to modify your /etc/apt/sources.list . This file should contain a bunch of `deb ... ` lines, followed by commented-out `deb-src` lines. Simply uncomment all `deb-src` lines ***iff*** they match the version (e.g. jammy, bionic, ...) of the `deb` lines. Otherwise, simply copy-paste each of the `deb` lines and add `-src` after the `deb`.

2. Install the following additionnal packages if you don't have them :
    * rustc compiler, ideally through [this](https://www.rust-lang.org/tools/install) 
    * `sudo apt install libbfd-dev libperl-dev libzstd-dev`


3. `sudo apt update`

4. Run the commands in the "Build Environment" section. 
5. Create a directory where the kernel source will reside, `cd` into it, and run the **apt** commande to get the sources.

## II. Approaching your first build

6. Important! Before your first build, add an unique identifier to your version number (that will appear when you run `uname -r` after booting your kernel). Do this by changing either (or both) the debian.master changelog, or the debian.hwe-*VERSION* changelog in the following manner: 

    `linux (6.8.0-45.45) jammy; urgency=medium` --> `linux (6.8.0-45+mykernel.45) jammy; urgency=medium`
    
    **Very importantly** (otherwise you will get a `dh_prep -p` "package not found" error at the very end of your build), also change the `/debian/changelog` in the same way.

    Note the placement of the my custom string ("+mykernel") -- it is **after** the `-ABI NUMBER` (`-45` here), and **before** the `.XX` (`.45` here).

    To know which version will be used, you can run (from the source kernel directory), `debian/rules printenv` (which eventually executes the `printenv` make function in `debian/rules.d/1-maintainer.mk` if you want to add other printed variables for example -- I added `	@echo "DEBIAN                    = $(DEBIAN)"` for instance)

    If you decide to add more than simply numbers as your unique identifier (as I did in the example with "+mykernel"), then Microsoft's Hyper-V will not build (it is parsing the release number as an integer compiler defined macro). See the point below for the fix.

6. "Edit" the configs `fakeroot debian/rules editconfigs`, make sure to say "y" to every edit proposition, and then simply exit the edit menuconfig (otherwise you might encounter some weird build issues afterwards). Note that it is normal if you get a bunch of "file not found" errors for configs of architecture your machines doesn't support -- you can safely ignore them.

7. Make your kernel changes ! For us, this means apply the [following patch](TODO), which does the following:
    1. Fixes the Hyper-V bug mentionned in 5 (by parsing the version numbers as a string, and extracting the first digits, assuming a "+" as a separator)
    2. ...

9. Finally, build your kernel. Make a full build, as you will need linux tools : `fakeroot debian/rules binary`. This build will take at least ~1 hour, depending on your machine's performance. Go take a coffee break :)

##  III. Installing and Booting off your kernel 

10. If there are no errors, you should have received, as output in the parent directory of the source of the kernel (where you built from), you should have obtained a bunch of `.deb` files -- simply install them in your `/boot` directory by running `sudo dpkg -i linux*.deb`.

10. If you're running this on a machine with physical access, simply restart it. In your grub menu should appear your new kernel (with your custom name prefix if you've set it in 5.). 
    
    If you're doing this on a remote server though (throug SSH), you will probably not (ever) have physical access to the machine : make sure to verify that the current `grub` boot order of your kernels has an existing, working kernel (e.g.: the current one you're building from) set as first boot choice and then reboot using `grub-reboot [OPTION] MENU_ENTRY` (see [this](https://askubuntu.com/questions/574295/how-can-i-get-grub2-to-boot-a-different-option-only-on-the-next-boot)) which boots in a specific kernel for the next boot only! This way, if some of your changes in 7. broke the kernel (e.g.: can't even boot properly), a simple reboot (from an admin console for example) should put it back to the previous, working kernel, and you will hopefully be able to examine what went wrong in the system logs.

# Encountered errors:

### mv: cannot stat 'debian/build/tools-perarch/tools/bpf/bpftool/vmlinux': No such file or directory

This error usually happens only if you the first build hasn't succeeded. There is a [bug report](https://bugs.launchpad.net/ubuntu/+source/linux/+bug/2047683) about it. A quick fix however is simply (from the source directory): `ln -s tools/bpf/bpftool/vmlinux debian/build/tools-perarch/tools/bpf/bpftool/vmlinux` and re-running the build command (which will now succeed -- don't worry it won't take as long, as all the kernel files have already been compiled).  