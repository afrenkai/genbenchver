#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dfox
Intended to be the main program, with many files for portions of the code.
Right now it's just one big script.
"""
import os
import pandas as pd
import numpy as np
import random
import time
import datetime as dt
import io
import itertools
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from nnsight import LanguageModel
import shutil
import math
import copy

from gbv_prompts import GenAITablePrompts
from gbv_ver_table import VerTable, VerTableCache
from gbv_utils import print_time

# Note only semi-colon-delimited csv files are supported at this time

# MODEL_SPEC = 'learnanything/llama-7b-huggingface'
MODEL_SPEC = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# MODEL_SPEC = 'mistralai/Mistral-7B-v0.1'

TABLES_VERSION_DELIMITER = "_"
FORMAT_CSV_DELIMITER = ';'
FORMAT_CSV_DELIMITER_NAME = "semi-colon"
FORMAT_CSV = '.csv'
FORMAT_TYPE = (FORMAT_CSV, FORMAT_CSV_DELIMITER, FORMAT_CSV_DELIMITER_NAME)
SUPPORTED_FORMATS = [FORMAT_CSV]

COMMANDS = [
    {'type': 'add_rows',
     'params': {'num_entries' : [1, 2, 3, 4, 5],
                'location' : ['top', 'bottom', 'random']}},
    {'type': 'del_rows',
     'params': {'num_entries' : [1, 2, 3, 4, 5],
                'location' : ['random']}},
    {'type': 'add_cols',
     'params': {'num_entries' : [1, 2, 3], # [1, 2, 3, 4, 5],
                'location' : ['left', 'right', 'random']}},
    {'type': 'del_cols',
     'params': {'num_entries' : [1, 2, 3, 4, 5],
                'location' : ['random']}},
    {'type': 'fill_na',
    'params': {'location' : ['first', 'random']}}
    ]

# CMD_PLAN = [
#     4, # fill_na
#     4, # fill_na
#     0, # add_row
#     2, # add_col
#     3, # del_col
#     1, # del_row
#     4, # fill_na
#     2, # add_col
#     1, # del_row
#     0, # add_row
#     'random'
#     ]

TIME_START = time.time()
DATETIME_START = dt.datetime.now()

random.seed(TIME_START)

def get_params_list(params_dict):
    """
    

    Parameters
    ----------
    params_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    keys = params_dict.keys()
    vals = params_dict.values()
    combinations = list(itertools.product(*vals))
    return [dict(zip(keys, combination)) for combination in combinations]

class GenAITableExec:
    # Singleton
 
    def __init__(self, cache, model, tokenizer):
        self.args = None
        self.tables_version_delimiter = TABLES_VERSION_DELIMITER
        # model_type = 'nnsight'
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        # self.model_spec = MODEL_SPEC
    
    def print_debug(self, var, text):
        if not self.args.debug:
            return
        print_time(var, text)

    def convert(self):
        fn = os.path.join(self.args.tablesdir, self.args.name + FORMAT_TYPE[0])
        df = pd.read_csv(fn, sep=self.args.orig_delim)
        fn = os.path.join(self.args.tablesdir, (self.args.name + "_0" 
                                                + FORMAT_TYPE[0]))
        df.to_csv(fn, sep=FORMAT_CSV_DELIMITER, index=False)
        
        sfn = os.path.join(self.args.tablesdir, self.args.name + ".json")
        dfn = os.path.join(self.args.tablesdir, self.args.name + "_0.json")
        shutil.copy(sfn, dfn)
    
    def execute_prompts(self, genai_prompts):
        """
        
    
        Parameters
        ----------
        model_type : TYPE
            DESCRIPTION.
        tokenizer : TYPE
            DESCRIPTION.
        model : TYPE
            DESCRIPTION.
        max_new_tokens : TYPE
            DESCRIPTION.
        prompts : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        random.seed(42)
        start_time = time.time()
        start = dt.datetime.now()
        print_time(f"--- starting at {start}", None)
        prompts_output = []
        
        if self.framework == 'nnsight':
            with self.model.generate(max_new_tokens=genai_prompts.max_new_tokens, 
                                     remote=False) as generator:
                print_time("--- %s seconds ---" % (time.time() - start_time), None)
                for prompt in genai_prompts.prompts:
                    with generator.invoke("[INST] " + prompt + " /[INST]"):
                        pass
                    print_time("finished with prompt", None)
                    print_time("--- %s seconds ---" % (time.time() - start_time), 
                               None)
            print_time("--- %s seconds ---" % (time.time() - start_time), None)
            model_response = self.tokenizer.batch_decode(generator.output)
            
            for i in range(len(genai_prompts.prompts)):
                prompts_output.append(model_response[i].split("</s")[0]\
                                      .split("/[INST]")[-1])
                
                
        
        elif self.framework == 'transformers':
            inputs = self.tokenizer("[INST] " + genai_prompts.prompts[0] + " /[INST]",
                                    return_tensors="pt").to(self.args.device_map)
            
            outputs = self.model.generate(
                **inputs, max_new_tokens=genai_prompts.max_new_tokens)
            for output in outputs:
                print_time("--- %s seconds ---" % (time.time() - start_time), None)
                model_response = self.tokenizer.decode(output,
                                                       skip_special_tokens=True)
                prompts_output.append(model_response.split("</s")[0]\
                                      .split("/[INST]")[-1])
    
        print_time("--- %s seconds ---" % (time.time() - start_time), None)
        return(prompts_output)
    
    def get_text_from_output(self, prompt_output, sep):
        preamble = prompt_output['preamble']
        table_df = prompt_output['output_table']
        output_table = table_df.to_csv(sep=sep, index=False)
        postamble = prompt_output['postamble']
        
        return table_df, preamble, output_table, postamble

    def add_table(self, table_orig, table_df, location, axis):
        self.print_debug(table_df, None)
        if table_df is None:
            print_time(None, "add_row or add_col: Table not found!")
            time.sleep(10)
            return None # we did not find a table in the response, do nothing
        
        df = table_df.copy()
        df.reset_index(drop=True, inplace=True)
        n_entries = table_orig.table.shape[axis]
        old_table = table_orig.table.copy()
        if axis == 0:
            for col in df:
                if col not in old_table:
                    old_table[col] = np.nan
            for col in old_table:
                if col not in df:
                    df[col] = np.nan
            self.print_debug(old_table, "row")
            self.print_debug(df, "row")
            if location == 0:
                new_df = pd.concat([df, old_table], axis=axis)
            elif location == (n_entries-1):
                new_df = pd.concat([old_table, df], axis=axis)
            else:
                before = old_table.head(location).copy()
                after = old_table.tail(n_entries - location).copy()
                new_df = pd.concat([before, df, after], axis=axis)
        elif axis == 1:
            # For column we do an outer join on the semantic key so as not to
            # use the index
            
            self.print_debug(df, "col")
            self.print_debug(old_table, "col")
            new_df = old_table.merge(df, how='outer', on=table_orig.semantic_key)
            self.print_debug(new_df, "col")
            
            cols_new = []

            if location == 0:
                
                for col in df:
                    if col not in table_orig.semantic_key:
                        cols_new.append(col)
                for col in old_table:
                    cols_new.append(col)

            elif location == (n_entries-1):
                
                for i, col in enumerate(old_table):
                    if i < location:
                        cols_new.append(col)
                for col in df:
                    if col not in table_orig.semantic_key:
                        cols_new.append(col)
                for i, col in enumerate(old_table):
                    if i >= location:
                        cols_new.append(col)
                        
            else:
                
                for col in old_table:
                    cols_new.append(col)
                for col in df:
                    if col not in table_orig.semantic_key:
                        cols_new.append(col)
                    
            new_df = new_df[cols_new]
            
        new_df.reset_index(drop=True, inplace=True)            
        self.print_debug(new_df, None)
        if new_df is None:
            print_time(None, "Bad csv format of output")
            time.sleep(10)
            return None
        new_df = new_df.drop_duplicates()
        return new_df
            
    def add_rows_exec(self, table_orig, params):
        num_entries = params['num_entries']
        location = params['location']
        # axis = 0
        nrows = table_orig.table.shape[0]
        if params['location'] == 'top':
            location = 0
        elif params['location'] == 'bottom':
            location = nrows - 1
        else: # 'random'
            location = random.randrange(nrows)
        params['location'] = location
        
        genai_prompts = GenAITablePrompts(self.cache, table_orig, 50000)
        genai_prompts.add_prompt('add_rows', nrows=num_entries)
        
        if self.args.framework == 'fake':
            responses = []
            responses.append(
                " Here are 2 new attributes generated for the table:"\
                + "1. FuelType: This attribute indicates the type of fuel used "\
                + "by the vehicle. The possible values are Gas, Diesel, Hybrid, "\
                + "and Electric. The data for this attribute is obtained from "\
                + "the official websites of the manufacturers or third-party "\
                + "automotive data providers.\n"\
                + "2. CityMPG: This attribute indicates the fuel efficiency "\
                + "of the vehicle in city driving conditions, "\
                + "measured in miles per gallon (MPG)."\
                + "The data for this attribute is obtained from the official "\
                + "fule economy ratings provided by the "\
                + "Environmental Protection Agency (EPA) of the United States."
                )
            resp_table = table_orig.table.copy()
            if resp_table.shape[0] == 0:
                # our table is empty, use the original table for a source of fake 
                # rows, note that the original table was required to have at least
                # one row
                prev_table = self.cache.add(table_orig.name, 0)
                was_none_prev = False
                if prev_table.table is None:
                    prev_table.read()
                    was_none_prev = True
                resp_table = prev_table.table.copy()
                if was_none_prev:
                    prev_table.purge()
            while resp_table.shape[0] < num_entries:
                resp_table = pd.concat([resp_table, resp_table], axis=0)
            head_nrows = resp_table.head(num_entries).to_csv(sep=table_orig.format_type[1],
                                                       index=False)
            responses.append(
                f"Preamble\n\n{head_nrows}\n\nPostamble\n"
                )
            responses.append(
                f"{head_nrows}\nPreamble\n\n{head_nrows}\n\nPostamble\n"
                )
            idx = random.randrange(len(responses))
            responses = responses[idx:idx+1]
            print_time(responses, None)
        else:
            responses = self.execute_prompts(genai_prompts)

        rsp =  self.parse_table_responses(table_orig, responses, num_entries) 

        return genai_prompts.prompts, rsp

    def add_rows(self, table_orig, params):
        start_time = round(time.time(), 0)
        axis = 0
        prompts_input, prompts_output = \
            self.add_rows_exec(table_orig, params)

        if prompts_output is None or len(prompts_output) == 0:
            print_time(None, "add_rows A : Table not found!")
            time.sleep(10)
            return None, None

        table_df, preamble, output_table, postamble = \
            self.get_text_from_output(prompts_output[0], 
                                      table_orig.format_type[1])
        if output_table == "":
            print_time(None, "Table not found!")
            time.sleep(10)
            return None, None # we did not find a table in the response, do nothing
            
        new_df = self.add_table(table_orig, table_df,
                                # prompts_output[0]['output_table'], 
                                params['location'], axis)
        if (table_df.shape[0] > table_orig.table.shape[0]):
            changed_df = pd.concat([table_df, new_df])
            changed_df = changed_df.drop_duplicates()
            self.print_debug(changed_df, "add_rows")
        else:
            changed_df = table_df.copy()
            self.print_debug(changed_df, "add_rows")

        self.print_debug(changed_df, None)
        end_time = round(time.time(), 0)
        duration = end_time - start_time
        if prompts_output is not None and len(prompts_output) > 0:
            output_table = prompts_output[0]['output_table']\
                .to_csv(sep=table_orig.format_type[1], index=False)
        else:
            output_table = None
            
        command_dict = {'type': 'add_rows',
                        'prompt': prompts_input[0],
                        'start time' : str(dt.datetime.fromtimestamp(
                            start_time)),
                        'complete time': str(dt.datetime.fromtimestamp(
                            end_time)),
                        'duration (seconds)': int(duration),
                        'params': params,
                        'output_table': output_table,
                        'preamble': preamble,
                        'postamble': postamble,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict
            
    def delete_rows(self, table_orig, params):
        num_entries = params['num_entries']
        location = params['location']
        nrows = table_orig.table.shape[0]
        num_entries = min(num_entries, nrows-3)
        if num_entries <= 0:
            return None, None
        if params['location'] == 'top':
            location = 0
        elif params['location'] == 'bottom':
            location = nrows - 1
        else:
            location = random.sample(list(range(nrows)), num_entries)
        df = table_orig.table.copy()
        print_time(location, None)
        changed_df = df.iloc[location,:].copy()
        idx = np.ones(len(df.index), dtype=bool)
        idx[location] = False
        new_df = df.iloc[idx].copy()
        command_dict = {'type': "del_rows",
                        'params': params,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict
    
    def add_cols_exec(self, table_orig, params):
        """
        
    
        Parameters
        ----------
        v_cache : TYPE
            DESCRIPTION.
        table_orig : TYPE
            DESCRIPTION.
        ncols : TYPE
            DESCRIPTION.
        model_type : TYPE
            DESCRIPTION.
        model : TYPE
            DESCRIPTION.
        tokenizer : TYPE
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        ncols = params['num_entries']
        genai_prompts = GenAITablePrompts(self.cache, table_orig, 50000)
        genai_prompts.add_prompt('add_cols', ncols=ncols)
        
        if self.args.framework == 'fake':
            old_table = table_orig.table.copy()
            resp_table = old_table[table_orig.semantic_key].copy()
            responses = []
            if old_table.shape[1] == 0:
                # our table is empty, use the original table for a source of fake 
                # rows, note that the original table was required to have at least
                # one row
                prev_table = self.cache.get(table_orig.name, 0)
                was_none_prev = False
                if prev_table.table is None:
                    prev_table.read()
                    was_none_prev = True
                old_table = prev_table.table.copy()
                if was_none_prev:
                    prev_table.purge()
            old_columns = list(old_table.columns)
            random.shuffle(old_columns)
            col_count = 0
            while col_count < ncols:
                for col in old_columns:
                    colnew = col + "_NEW"
                    if not col.endswith("_NEW") and not colnew in old_columns:
                        print_time(old_table, None)
                        print_time(col, None)
                        resp_table[colnew] = old_table[col]
                        print_time(resp_table, None)
                        col_count += 1
                        if col_count >= ncols:
                            break
                if col_count >= ncols:
                    continue
                for col in old_columns:
                    colnew = col + "_NEW"
                    if not colnew in old_columns:
                        resp_table[colnew] = old_table[col]
                        print_time(resp_table, None)
                        col_count += 1
                        if col_count >= ncols:
                            break
                if col_count >= ncols:
                    continue
                for col in old_columns:
                    resp_table[col + "_NEW"] = old_table[col]
                    print_time(resp_table, None)
                    col_count += 1
                    if col_count >= ncols:
                        break
                    
                        
            table_str = resp_table.to_csv(sep=table_orig.format_type[1], index=False)
            responses.append(
                f"Preamble\n\n{table_str}\n\nPostamble\n"
                )
            responses.append(
                f"{table_str}\nPreamble\n\n{table_str}\n\nPostamble\n"
                
                )
            idx = random.randrange(len(responses))
            responses = responses[idx:idx+1]
        else:
                
            responses = self.execute_prompts(genai_prompts)
            # responses = execute_prompts_static(model_type, tokenizer, model,
            #                                    device_map,
            #                                    genai_prompts)

        rsp =  self.parse_table_responses(
            table_orig, responses, table_orig.table.shape[0]) 
    
        return genai_prompts.prompts, rsp
    
    def add_cols(self, table_orig, params):
        start_time = round(time.time(), 0)
        ncols = params['num_entries']
        location = params['location']
        axis = 1
        ncols = table_orig.table.shape[1]
        if params['location'] == 'left':
            location = 0
        elif params['location'] == 'right':
            location = ncols - 1
        else:
            location = random.randrange(ncols)
        prompts_input, prompts_output = self.add_cols_exec(table_orig,
                                                           params)
        if (prompts_output is None or prompts_input is None 
            or len(prompts_output) == 0):
            print_time(None, "add_col: Table not found!")
            time.sleep(10)
            return None, None
        table_df, preamble, output_table, postamble = \
            self.get_text_from_output(prompts_output[0], 
                                      table_orig.format_type[1])
        if output_table == "":
            print_time(None, "Table not found!")
            time.sleep(10)
            return None, None # we did not find a table in the response, do nothing
            
        new_df = self.add_table(table_orig, table_df,
                                # prompts_output[0]['output_table'], 
                                location, axis)
        changed_df = table_df.copy()
        for col in table_orig.semantic_key:
            if col in changed_df:
                changed_df = changed_df.drop(col, axis=1)
        begin_time = round(start_time, 0)
        end_time = round(time.time(), 0)
        duration = end_time - begin_time
        if len(prompts_output) > 0:
            output_table = prompts_output[0]['output_table']\
                .to_csv(sep=table_orig.format_type[1], index=False)
        else:
            output_table = None
        command_dict = {'type': "add_cols",
                        'prompt': prompts_input[0],
                        'start time' : str(dt.datetime.fromtimestamp(
                            begin_time)),
                        'complete time': str(dt.datetime.fromtimestamp(
                            end_time)),
                        'duration (seconds)': int(duration),
                        'params': params,
                        'output_table': output_table,
                        'preamble': preamble,
                        'postamble': postamble,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict

    def del_cols(self, table_orig, params):
        num_entries = params['num_entries']
        location = params['location']
        indices_elig = []
        print("")
        print("table_orig.table")
        self.print_debug(table_orig.table, None)
        for i, col in enumerate(table_orig.table):
            if col not in table_orig.semantic_key:
                indices_elig.append(i)
        ncols = table_orig.table.shape[1]
        max_indices_elig = math.floor(len(indices_elig) / 2)
        if num_entries > max_indices_elig:
            num_entries = max_indices_elig
            if num_entries > 0:
                print_time("del_col: reduced number of columns to delete "
                           + f"outside of semantic key to {num_entries} "
                           + "because will not allow to delete more than "
                           + "half (rounded down) of the columns outside of "
                           + "the semantic key", None)
            else:
                print_time("del_col: delete columns cancelled because will "
                           + "not allow to delete more than half "
                           + "(rounded down) of the columns outside of the "
                           + "semantic key", None)
                time.sleep(10)
                return None, None
        if params['location'] == 'left':
            location = 0
        elif params['location'] == 'right':
            location = ncols - 1
        else: # params['location'] == 'random'
            self.print_debug(indices_elig, None)
            self.print_debug(num_entries, None)
            location = random.sample(indices_elig, num_entries)
            self.print_debug(location, None)
        params['location'] = location
        
        df = table_orig.table.copy()
        
        print_time(location, None)
        col_del_list = []
        for i, col in enumerate(df):
            if i in location:
                col_del_list.append(col)
                print_time(None, f"deleting col {col} for location {i}")
                print_time(col_del_list, None)
                
        print_time(location, None)
        print_time(df, None)
        changed_df = pd.DataFrame(columns=col_del_list)
        for i, col in enumerate(df):
            if i in location:
                # col_del_list.append(col)
                print_time(None, f"deleting col {col} for location {i}")
                print_time(changed_df, None)
                
                changed_df[col] = df[col]
        new_df = df.drop(df.columns[location], axis=1)
        
        self.print_debug(new_df, None)
        command_dict = {'type': "del_cols",
                        'params': params,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict

    def fill_na_exec(self, table_orig, params):
    
        na_loc = params['na_loc']
                
        genai_prompts = GenAITablePrompts(self.cache, table_orig, 50000)
        genai_prompts.add_prompt('fill_na', na_loc=na_loc)
        
        if self.args.framework == 'fake':
            was_none = False
            was_none = False
            if table_orig.table is None:
                was_none = True
                table_orig.read()
            resp_table = table_orig.table.copy()
            sem_key = table_orig.semantic_key
            sem_val = []
    
            for col in sem_key:
                sem_val.append(table_orig.table.at[na_loc[1], col])
            num_rows = min(table_orig.table.shape[0], 3)
            if num_rows == 3:
                if na_loc[0] == 0:
                    rows = [na_loc[0], 1, 2]
                elif na_loc[0] == 1:
                    rows = [na_loc[0], 0, 2]
                else:
                    rows = [na_loc[0], 0, 1]
            elif num_rows == 2:
                if na_loc[0] == 0:
                    rows = [na_loc[0], 1]
                else:
                    rows = [na_loc[0], 0]
            elif num_rows == 1:
                rows = [na_loc[0]]
            attribute = na_loc[2]
            col_dtype = str(table_orig.table[attribute].dtype)
    
            if na_loc[0] == 0:
                if resp_table.shape[0] > 1:
                    dummy_val = resp_table.iloc[resp_table.index[1], na_loc[2]]
                else:
                    dummy_val = 0
            else:
                dummy_val = resp_table.at[resp_table.index[0], na_loc[2]]
            resp_table.at[na_loc[1], na_loc[2]] = dummy_val
            small_table = resp_table.iloc[rows,:].to_csv(
                sep=table_orig.format_type[1], index=False)
            attribute = na_loc[2]
            col_dtype = str(resp_table[attribute].dtype)
            responses = []
            responses.append(
                f"The missing {na_loc[2]} value for {sem_key} = {sem_val}"\
                + "in the table can be found on the official website. "\
                + f"According to the website, {na_loc[2]} is {dummy_val}. "\
                + f"To fill in the missing data with a dtype of {col_dtype}, "\
                + f"we can convert the value to {col_dtype} using the numpy "\
                + "library. "\
                + f"Here's the completed table with the missing {na_loc[2]} "\
                + f"value filled in as {col_dtype}:\n\n{small_table}\n\n"\
                + "And here's the resulting table in "\
                + f"semi-colon-delimited .csv format:\n\n{small_table}\n\n"\
                + f"The missing {na_loc[2]} value of {dummy_val} was retrieved "\
                + "from the official website.")
            responses.append(
                f"Preamble\n\n{small_table}\n\nPostamble\n"
                )
            responses.append(
                f"{small_table}\nPreamble\n\n{small_table}\n\nPostamble\n"
                
                )
            no_head_small_table = resp_table.iloc[rows,:].to_csv(
                sep=table_orig.format_type[1], index=False, header=None)
            responses.append(
                f"Preamble\n\n{no_head_small_table}\n\nPostamble\n"
                )
            responses.append(
                f"{no_head_small_table}\nPreamble\n\n{no_head_small_table}\n\nPostamble\n"
                )
            responses.append(
                f"{small_table}\nPreamble\n\n{no_head_small_table}\n\nPostamble\n"
                )
            small_lines = small_table.split('\n')
            if len(small_lines) > 3:
                small_lines[2] += ";;;;;;;;;;;;"
                small_lines[3] += ";;;;;;;;;;;;"
            x_table = '\n'.join(small_lines)
            responses.append(
                f"{x_table}\nPreamble\n\n{x_table}\n\nPostamble\n"
                
                )
            idx = random.randrange(len(responses))
            # idx = len(responses)-1
            responses = responses[idx:idx+1]
            
        else:
                
            responses = self.execute_prompts(genai_prompts)
            
        rsp =  self.parse_table_responses(
            table_orig, responses, min(table_orig.table.shape[0], 3)) 
    
        if was_none:
            table_orig.purge()
            
        return genai_prompts.prompts, rsp
    
    def find_all_na(self, df):
        na_df = df.isna().copy()
        self.print_debug(na_df, None)
        na_indices = []
        i = 0
        for index, row in na_df.iterrows():
            for col in na_df:
                if na_df.at[index, col]:
                    na_indices.append((i, index, col))
            i += 1
        return na_indices
    
    def fill_na(self, table_orig, params):
        start_time = round(time.time(), 0)
        # location = params['location']
        na_loc = None
        na_list = self.find_all_na(table_orig.table)
        if len(na_list) == 0:
            print_time(None, "fill_na: no N/A values to retrieve!")
            time.sleep(10)
            return None, None
        if params['location'] == 'random':
            random.shuffle(na_list)
        na_loc = na_list[0]
        params['na_loc'] = na_loc
        
        prompts_input, prompts_output = self.fill_na_exec(table_orig, params)

        if (prompts_input is None or len(prompts_input) == 0 or
            prompts_output is None or len(prompts_output) == 0):
            print_time(None, "fill_na: Table not found!")
            time.sleep(10)
            return None, None
        
        table_df, preamble, output_table, postamble = \
            self.get_text_from_output(prompts_output[0], 
                                      table_orig.format_type[1])
        if len(output_table) == 0:
            print_time(None, "Table not found!")
            time.sleep(10)
            return None, None # we did not find a table in the response, do nothing

        new_df = table_orig.table.copy()
        if na_loc[2] not in table_df:
            new_df.at[na_loc[1], na_loc[2]] = \
                table_df.at[table_df.index[0], 
                            new_df.columns.get_loc(na_loc[2])]
        else:
            new_df.at[na_loc[1], na_loc[2]] = table_df.at[table_df.index[0], 
                                                          na_loc[2]]
        
        changed_df = pd.DataFrame(columns=[na_loc[2]])
        changed_df.at[na_loc[1], na_loc[2]] = new_df.at[na_loc[1], na_loc[2]]
        
        print_time(changed_df, None)
        begin_time = round(start_time, 0)
        end_time = round(time.time(), 0)
        duration = end_time - begin_time
        if prompts_output is not None and len(prompts_output) > 0:
            output_table = prompts_output[0]['output_table']\
                .to_csv(sep=table_orig.format_type[1], index=False)
        else:
            output_table = None
        command_dict = {'type': "fill_na",
                        'prompt': prompts_input[0],
                        'start time' : str(dt.datetime.fromtimestamp(
                            begin_time)),
                        'complete time': str(dt.datetime.fromtimestamp(
                            end_time)),
                        'duration (seconds)': int(duration),
                        'params': params,
                        'output_table': output_table,
                        'preamble': preamble,
                        'postamble': postamble,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict
    
    def new_table(self, table_orig, new_df, command_dict, new_version):

        """
        
    
        Parameters
        ----------
        cache : TYPE
            DESCRIPTION.
        table_orig : TYPE
            DESCRIPTION.
        preamble : TYPE
            DESCRIPTION.
        new_df : TYPE
            DESCRIPTION.
        postamble : TYPE
            DESCRIPTION.
        command_dict : TYPE
            DESCRIPTION.
        new_version : TYPE
            DESCRIPTION.
    
        Returns
        -------
        new_table : TYPE
            DESCRIPTION.
    
        """
        lineage = copy.deepcopy(table_orig.lineage)
        self.print_debug(lineage, "E")
        if lineage is None:
            lineage = []
        # if table_orig.version > 0:
        #     if len(lineage) > 0:
        #         assert(lineage[-1] != table_orig.version)
        #     lineage.append(table_orig.version)
        if len(lineage) > 0:
            assert(lineage[-1] != table_orig.version)
        lineage.append(table_orig.version)
        self.print_debug(lineage, "A")
        info = {'description': table_orig.description,
                'lineage': lineage,
                'semantic_key': table_orig.semantic_key,
                'file_ext': table_orig.format_type[0],
                'field_delim': table_orig.format_type[1],
                'file_ext_name': table_orig.format_type[2]}
        new_table = VerTable(new_df, table_orig.name, table_orig.folder,
                             new_version, info, 
                             version_delimiter=self.tables_version_delimiter,
                             format_type=table_orig.format_type,
                             create_command=command_dict,
                             debug=self.args.debug)
        self.cache.add(new_table)
        
    def find_valid_csv_tables(self, table_orig, text, nrows_expected):
        valid_tables = []
        cols_expected = table_orig.semantic_key
        sep = table_orig.format_type[1]
        
        lines = text.split("\n")
        if len(lines) < nrows_expected:
            return valid_tables
        hdridx = []
        rowidx = []
        for i in range(len(lines)):
            remain = len(lines) - i
            if remain >= nrows_expected:
                found_cols = True
                for col in cols_expected:
                    if col not in lines[i]:
                        found_cols = False
                        break
                if found_cols:
                    hdridx.append(i)
                else:
                    row = lines[i].split(sep)
                    if len(row) >= len(cols_expected):
                        rowidx.append(i)
            else:
                break
            
        for i in rowidx:
            if i > 0:
                j = i-1
                if j in hdridx:
                    continue
            valid = True
            for j in range(i, i+nrows_expected):
                if j not in rowidx:
                    valid = False
            if valid:        
                table_text = "\n".join(lines[i:i+nrows_expected])
                self.print_debug(table_text, None)
                try:
                    df = pd.read_csv(io.StringIO(table_text), sep=sep, header=None,
                                     names=list(table_orig.table.columns))
                    preamble_text = "\n".join(lines[0:i])
                    postamble_text = "\n".join(lines[i+nrows_expected:])
                    valid_table = (preamble_text, df.copy(), postamble_text)
                    valid_tables.append(valid_table)
                except:
                    continue
            
        for i in hdridx:
            valid = True
            hdrlen = len(lines[i].split(sep))
            for line_idx in range(i, i+nrows_expected+1):
                fieldslen = len(lines[line_idx].split(sep))
                if fieldslen < len(cols_expected):
                    valid = False
                    break
                elif fieldslen > hdrlen:
                    fields = lines[line_idx].split(sep)
                    lines[line_idx] = ';'.join(fields[:hdrlen])
            
            if valid:
                for j in range(len(lines)):
                    self.print_debug(lines[j], None)
                    lines[j] = lines[j].strip()
                    self.print_debug(lines[j], None)
                table_text = "\n".join(lines[i:i+nrows_expected+1])
                self.print_debug(table_text, None)
                try:
                    df = pd.read_csv(io.StringIO(table_text), sep=sep)
                    preamble_text = "\n".join(lines[0:i])
                    postamble_text = "\n".join(lines[i+nrows_expected+1:])
                    valid_table = (preamble_text, df.copy(), postamble_text)
                    valid_tables.append(valid_table)
                except:
                    continue
    
        return valid_tables
    
    def parse_table_responses(self, table, responses, nrows_expected):
        """
        
    
        Parameters
        ----------
        table : TYPE
            DESCRIPTION.
        response : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        # only supports one response for now
        table_responses = []
        self.print_debug(nrows_expected, None)
        for response in responses:
            
            self.print_debug(response, None)
            # time.sleep(10)
            
            # BEGIN SEARCH FOR TABLE WITHIN RESPONSE        
            # The first thing we need to do is locate our table within the
            # prompt response.
            # We have to make some assumptions here.
            #   1. Search for a table that conforms to what we asked for.
            #   2. If there's more than one, *assume it's the final table output.*
            #      This is a reasonable assumption. The chances of the table
            #      being within the prologue are far higher than the chances
            #      of the table being within the epilogue
    
            valid_csv_tables = self.find_valid_csv_tables(table, response,
                                                          nrows_expected)
            self.print_debug(valid_csv_tables, None)
    
            # loop through valid tables
            col_valid_csv_table = None
            for csv_table in valid_csv_tables:
                col_valid_csv_table = csv_table # look for the final table
                
            if col_valid_csv_table is not None:
                table_response = {
                    'preamble': col_valid_csv_table[0],
                    'output_table': col_valid_csv_table[1].copy(),
                    'postamble': col_valid_csv_table[2]
                    }
                self.print_debug(table_response, None)
                table_responses.append(table_response)
                
        return(table_responses)
    
    def command_exec(self, cmd_id):
        # start_time = time.time()
        
        cache = self.cache
        table_name = self.args.name
        
        new_version = cache.get_next_ver_for_table(table_name)
        self.print_debug(new_version, None)
        if self.args.lineage == "random":
            table_orig = cache.get_table_random_from_cache(table_name)
        else: # "sequence"
            table_orig = cache.get(table_name, 
                                   cache.get_high_ver_for_table(table_name))
        self.print_debug(table_orig.version, None)
        self.print_debug(table_orig.semantic_key, None)
        assert(len(table_orig.semantic_key) > 0)
        if table_orig.table is None:
            was_none = True
            table_orig.read()
        else:
            was_none = False
        # prompts_input = None
        # prompts_output = None
        command = COMMANDS[cmd_id]
        self.print_debug(table_orig.semantic_key, None)
        command_type = command['type']
        print_time(command, f"Starting operation...{command_type}")
        params = command['params']
        params_list = get_params_list(params)
        random.shuffle(params_list)
        params = params_list[0]
        
        if command_type == "add_rows":
            new_df, command_dict = self.add_rows(table_orig, params)
        elif command_type == "del_rows":
            new_df, command_dict = self.delete_rows(table_orig, params)
        elif command_type == "add_cols":
            new_df, command_dict = self.add_cols(table_orig, params)
        elif command_type == "del_cols":
            new_df, command_dict = self.del_cols(table_orig, params)
        elif command_type == "fill_na":
            new_df, command_dict = self.fill_na(table_orig, params)
    
        if new_df is None:
            return False
        
        self.new_table(table_orig, new_df, command_dict, new_version)

        if was_none:
            table_orig.purge()

        print_time(command, "Finished successfully")
        return True
        
    def main(self, args):
        
        self.args = args

        numver = 0
        for _ in range(self.args.max_iter):
            command_idx = random.randrange(len(COMMANDS))
            if self.command_exec(command_idx):
                numver += 1
            if numver >= self.args.num_ver:
                break

def build_model(framework, model_id, device_map, data_type):
    """
    

    Parameters
    ----------
    model_type : TYPE
        DESCRIPTION.
    model_id : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if framework == 'fake':
        return None, None
    model_id = MODEL_SPEC
    
    if framework == 'nnsight':
        tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                                       unk_token="<unk>",
                                                       pad_token='[PAD]')
        model = LanguageModel(model_id, device_map=device_map, 
                                   tokenizer=tokenizer)
        
    elif framework == 'transformers':
        # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        if data_type == "float16":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map, torch_dtype=torch.float16)
        else: # None
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map) #, torch_dtype=torch.float16)

        # if ENVIRONMENT == "turing.wpi.edu":
        #     model = AutoModelForCausalLM.from_pretrained(
        #         model_id, device_map='auto') #, torch_dtype=torch.float16)
        # else: # ENVIRONMENT == "local_macos"
        #     model = AutoModelForCausalLM.from_pretrained(
        #         model_id, device_map='auto', torch_dtype=torch.float16)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, unk_token="<unk>")

    return model, tokenizer

# """
# Begin Script
# """

def main():
    print_time(None, "Starting globally...")
    
    parser = argparse.ArgumentParser(
        description=('Auto-generates versions of base file using '
                     + 'generative AI'))
    parser.add_argument(
        '--debug', dest='debug', action='store_true',default=False,
        help='Turns on debug logging to stdout. Default is off.')
    # parser.add_argument(
    #     '--fake', dest='fake_model', action='store_true', 
    #     default=False,
    #     help=('Tests code with fake responses without using the '
    #           + 'generative AI. Default uses real interaction.'))
    parser.add_argument(
        '-s', '--sep', dest='orig_delim', type=str, default=',',
        help=('Column separator of the original file. Only used before '
              + 'versions have been generated. Default is comma'))
    parser.add_argument(
        '-l', '--lineage', dest='lineage', type=str, default="sequence",
        help=('Type of lineage for versions created: '
              + '"sequence" (default) | "random"'))
    parser.add_argument(
        '-g', '--gpu', dest='device_map', type=str, default="auto",
        help=('Specify type of GPU: "cuda" (default) | "mps". '
              + 'Default is "auto"'))
    parser.add_argument(
        '-n', '--numver', dest='num_ver', type=int, default=10,
        help=('Number of versions to create. Default is 10. '
              + 'Unsuccessful attempts do not count'))
    parser.add_argument(
        '-i', '--maxiter', dest='max_iter', type=int, default=20,
        help='Maximum number of table create attempts. Default is 20.')
    parser.add_argument(
        '-f', '--framework', dest='framework', type=str, default='nnsight',
        help=('Model framework type: nnsight | transformers | fake. '
              + 'Default is nnsight'))
    parser.add_argument(
        '-d', '--datatype', dest='dtype', type=str, default=None,
        help=('Data type for model. Use "float16" to reduce footprint. '
              + ' Default is None'))
    parser.add_argument('name', type=str,
        help=('Filename of the table without extension '
              + '(only .csv extension is supported)'))
    parser.add_argument('tablesdir', type=str, default=".",
        help=('Directory locaton of the tables. ' 
              + 'Default is the current working directory.'))
    
    args = parser.parse_args()
    
    model, tokenizer = build_model(args.framework, MODEL_SPEC, 
                                   args.device_map, args.dtype)
    v_cache = VerTableCache(args.tablesdir, 
                            supported_formats=SUPPORTED_FORMATS,
                            debug=args.debug, new_format_type=FORMAT_TYPE)
    genaitable_exec = GenAITableExec(v_cache, model, tokenizer)
    genaitable_exec.main(args)
    
if __name__ == '__main__':
    main()
    
"""
End script
"""