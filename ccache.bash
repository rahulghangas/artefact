#!/bin/bash
apt-get update
apt-get install -qq ccache
export PATH="/usr/lib/ccache:$PATH"

# Setup path for ccache
export CCACHE_BASEDIR=${PWD}
export CCACHE_DIR=${PWD}/ccache

# You need to add this to your GitLab CI config file:
#cache:
#  paths:
#    - ccache/

