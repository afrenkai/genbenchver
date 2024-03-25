#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:38:53 2024

@author: dfox
"""
import datetime as dt
import time
import itertools
import inspect

def get_variable_name(var):
    current_frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(current_frame)
    for i in range(len(outer_frames)-1, 0, -1):
        caller_frame = outer_frames[i]
        local_vars = caller_frame.frame.f_locals
     
        for name, value in local_vars.items():
            if value is var:
                return name
        for name, val in globals().items():
            if val is var:
                return name
    return None

def print_time(var, text):
    fmt_text = f"\n{dt.datetime.now()}\n"
    # fmt_text = f"\n{dt.datetime.now()}, {time.time() - TIME_START} seconds\n"
    if var is not None:
        varstr = get_variable_name(var)
        if varstr is not None and varstr != "var":
            fmt_text += f"{get_variable_name(var)} =\n{var}\n"
        else:
            fmt_text += f"\n{var}\n"
    if text is not None:
        fmt_text += f"{text}\n"
    print(fmt_text, flush=True)

# def print_time(var, text):
#     if not DEBUG:
#         return
#     print_time_force(var, text)
