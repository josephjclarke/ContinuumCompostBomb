#!/usr/bin/env bash

for file in figure*.py; do
    echo "==========================="
    echo $file
    echo "==========================="
    python $file
done
