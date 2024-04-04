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

# input_filename = '../../tables/sources/Horticulture_XDG29OB3.csv'
# cleaned_filename = '../../tables/Horticulture.csv'
# sep = "|"
# newsep = ";"

input_filename = '../../tables/sources/Climatology_QWQ27CT2.csv'
cleaned_filename = '../../tables/Climatology.csv'
sep = "|"
newsep = ";"

# Remove leading and trailing whitespaces around the pipe separator
with open(input_filename, 'r') as infile, open(cleaned_filename, 'w') as outfile:
    for line in infile:
        cleaned_line = newsep.join(segment.strip() for segment in line.split(sep)) + '\n'
        outfile.write(cleaned_line)
