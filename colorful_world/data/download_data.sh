#!/usr/bin/env bash

wget "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
tar xzf lfw.tgz
rm lfw.tgz
python3 clean_lfw.py