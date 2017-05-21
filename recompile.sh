#!/bin/bash
rm ./likelihood.c
rm ./likelihood.html
rm ./likelihood.so
cython -a ./likelihood.pyx
gcc -shared -pthread -fPIC `python-config --cflags` -o likelihood.so likelihood.c
