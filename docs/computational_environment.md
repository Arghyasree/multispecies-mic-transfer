# Computational Environment

This document records the main software and hardware environment used for the final experiments.

Private information such as hostnames, usernames, IP addresses, and local filesystem paths is not included.

## Operating System

```text
Linux-7.0.0-28-generic-x86_64-with-glibc2.39
```

## CPU

```text
Architecture:                            x86_64
CPU(s):                                  32
On-line CPU(s) list:                     0-31
Model name:                              Intel(R) Core(TM) i9-14900KS
Thread(s) per core:                      2
Core(s) per socket:                      24
Socket(s):                               1
CPU(s) scaling MHz:                      15%
NUMA node0 CPU(s):                       0-31
```

## System Memory

```text
total        used        free      shared  buff/cache   available
Mem:           125Gi       5.7Gi       1.1Gi       4.8Mi       119Gi       119Gi
Swap:           46Gi       916Ki        46Gi
```

## GPU

```text
NVIDIA RTX PRO 5000 Blackwell, 595.71.05, 48935 MiB
```

## Python

The recorded experiment environment used:

```text
3.14.3 | packaged by Anaconda, Inc. | (main, Feb 24 2026, 22:51:43) [GCC 14.3.0]
```

The released code supports Python 3.11 or newer, as described in the main README.

## PyTorch and CUDA

```text
PyTorch: 2.10.0+cu130
CUDA runtime reported by PyTorch: 13.0
CUDA available: True
CUDA device: NVIDIA RTX PRO 5000 Blackwell
```

## Python Dependencies

The dependency file used by the repository is:

`requirements.txt`

The package versions in this file are pinned for reproducibility of the released evaluation code.

The portable dependency file uses:

`torch==2.10.0`

The final experiments used:

`torch==2.10.0+cu130`

PyTorch builds depend on the operating system, hardware, and CUDA installation. Users on another platform should install a compatible PyTorch build using the official PyTorch installation instructions while keeping the other dependency versions listed in `requirements.txt`.

## Reproducing the Environment

An identical GPU or operating system is not required to inspect the released benchmark, configurations, split definitions, feature matrices, or aggregate results.

To rerun the final evaluation, install the repository dependencies and use a PyTorch build that is compatible with the available hardware.

The main execution instructions are provided in:

- `README.md`
- `docs/reproducibility.md`
- `docs/execution_map.md`
