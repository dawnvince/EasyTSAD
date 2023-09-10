#!/bin/bash

args = (0 1 2 3 4 01 012 0123 01234 12 123 1234 23 234 34)
for i in args
do
    python utils/todsGenerator/unigenerator.py --type $i
done