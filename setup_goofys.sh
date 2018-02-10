#! /bin/bash
# Place goofys in your system path (e.g. /usr/bin/goofys)
mkdir -p s3bucket_goofys
goofys --debug_fuse dse-cohort3-group5 s3bucket_goofys

