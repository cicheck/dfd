#!/usr/bin/env bash

sphinx-apidoc -o . ../src/dfd
make html
