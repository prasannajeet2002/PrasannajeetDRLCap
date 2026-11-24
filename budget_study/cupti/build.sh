#!/bin/sh
# Build the CUPTI injection libraries (CUDA 12.3; no root needed).
C=/usr/local/cuda-12.3/targets/x86_64-linux
cd "$(dirname "$0")"
for n in kernel_trace kernel_stream; do
  gcc -O2 -fPIC -shared -o lib$n.so $n.c -I$C/include -L$C/lib -lcupti -lpthread -Wl,-rpath,$C/lib -Wall
done
