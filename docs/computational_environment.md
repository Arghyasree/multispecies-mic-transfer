# Computational environment

This file records the principal software and hardware environment used for the final experiments. Hostnames, usernames, IP addresses, and private filesystem locations are intentionally omitted.

## Operating system

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

## System memory

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

```text
3.14.3 | packaged by Anaconda, Inc. | (main, Feb 24 2026, 22:51:43) [GCC 14.3.0]
```

## PyTorch and CUDA

```text
PyTorch: 2.10.0+cu130
CUDA runtime reported by PyTorch: 13.0
CUDA available: True
CUDA device: NVIDIA RTX PRO 5000 Blackwell
```

## Pinned Python dependencies

The single canonical Python dependency manifest is
[`requirements.txt`](../requirements.txt). The package versions are pinned for
the released final evaluation. The exact experiment environment used
`torch==2.10.0+cu130`, as recorded above; users on another platform should
install the compatible PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/) while retaining the other
pins.
