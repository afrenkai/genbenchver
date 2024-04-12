#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:36:18 2024

@author: dfox

This script converts a table into the format required for generating
versioning benchmarks
"""

# input_filename = '../../tables/Literature_7U98ZTPF.csv'
# cleaned_filename = '../../tables/Literature.csv'
# sep = "|"
# newsep = ";"
# ignore_prefix = None

# input_filename = '../../tables/sources/Horticulture_XDG29OB3.csv'
# cleaned_filename = '../../tables/Horticulture.csv'
# sep = "|"
# newsep = ";"
# ignore_prefix = None

# input_filename = '../../tables/sources/Climatology_QWQ27CT2.csv'
# cleaned_filename = '../../tables/Climatology.csv'
# sep = "|"
# newsep = ";"
# ignore_prefix = None

# input_filename = '../../tables/sources/Biology_6RCZXNPM.csv'
# cleaned_filename = '../../tables/Biology.csv'
# sep = "|"
# newsep = ";"
# ignore_prefix = None

input_filename = '../../tables/sources/Mythology_5IDTY430.csv'
cleaned_filename = '../../tables/Mythology.csv'
sep = "|"
newsep = ";"
ignore_prefix = "-"


# Remove leading and trailing whitespaces around the separator
with open(input_filename, 'r') as infile, open(cleaned_filename, 'w') as outfile:
    for line in infile:
        cleaned = newsep.join(
            segment.strip().lstrip(ignore_prefix) \
                for segment in line.split(sep)).lstrip(newsep)
        if len(cleaned) > 0:
            cleaned_line = cleaned + '\n'
            outfile.write(cleaned_line)
