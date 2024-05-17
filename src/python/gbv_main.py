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
import sys

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
     'params': {'num_entries' : [1, 2, 3],
                'location' : ['top', 'bottom', 'random']}},
    {'type': 'del_rows',
     'params': {'num_entries' : [1],
                'location' : ['random']}},
    {'type': 'add_cols',
     'params': {'num_entries' : [1, 2, 3], # [1, 2, 3, 4, 5],
                'location' : ['left', 'right', 'random']}},
    {'type': 'del_cols',
     'params': {'num_entries' : [1],
                'location' : ['random']}},
    # {'type': 'fill_na',
    # 'params': {'location' : ['first', 'random']}},
    {'type': 'update_val',
    'params': {}}
    ]

# fill this out if you want a script to follow
# once the script, if any, is exhausted
# then a command is selected at random

CMD_PLAN = [
    # 0, # add_row
    # 2, # add_col
    # 4, # update_val
    # 3, # del_col
    # 1, # del_row
    # 2, # add_col
    # 0, # add_row
    # 5, # update_val
    # 0, # add_row
    # 2, # add_col
    # 5, # update_val
    # 3, # del_col
    # 1, # del_row
    # 2, # add_col
    # 0, # add_row
    # 5, # update_val
    # 0, # add_row
    # 2, # add_col
    # 5, # update_val
    # 3, # del_col
    # 1, # del_row
    # 2, # add_col
    # 0, # add_row
    ]

TIME_START = time.time()
DATETIME_START = dt.datetime.now()

random.seed(TIME_START)

def get_params_list(params_dict):
    keys = params_dict.keys()
    vals = params_dict.values()
    combinations = list(itertools.product(*vals))
    return [dict(zip(keys, combination)) for combination in combinations]

class GenAITableExec:
    """
    Singleton class for executing the creation of versions of tables
    """
 
    def __init__(self, cache, model, tokenizer):
        """
        Instantiates the GenATTableExec singleton.

        Parameters
        ----------
        cache : dict
            Cache of tables and their versions.
            { <str: table name>: { <int: version>: {'table': <VerTable: table>
                                                    }
                                  }
             }
            See classes VerTable, VerTableCache
            Note that a VerTable object does not have to have its actual table
            populated, so the cache is not populated with the full table of
            every version of every table.
        model : deep learning model
            The deep learning model that is built. Can be thought of as an 
            opaque handle.
        tokenizer : LLM model tokenizer
            The tokenizer for an LLM. Can be thought of as an opaque handle.

        Returns
        -------
        None.

        """
        self.args = None
        self.tables_version_delimiter = TABLES_VERSION_DELIMITER
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
    
    def print_debug(self, var, text):
        if not self.args.debug:
            return
        print_time(var, text)

    def execute_prompts(self, genai_prompts):
        """
        Sends a prompt to a LLM and returns the response. Note that special
        tokens are added to the input prompts such that only the text of the
        prompt should be specified. Also special tokens are filtered out in 
        the response as well.

        Parameters
        ----------
        genai_prompts : GenAITablePrompts
            Prompts built by one of the methods in GenAITablePrompts.
            Which method used depends on the command being executed.
            (Note that although multiple prompts can be supplied, only
             one prompt is supported at this time.)

        Returns
        -------
        prompts_output : List of str
            The output from the LLM in response to the prompts given.
            (Again, only one response to the first prompt is supported at this
             time.)

        """
        start_time = time.time()
        start = dt.datetime.now()
        print_time(f"--- starting at {start}", None)
        prompts_output = []
        
        if self.args.framework == 'nnsight':
            with self.model.generate(max_new_tokens=genai_prompts.max_new_tokens,
                                     do_sample=True, temperature=0.1,
                                     remote=False) as generator:
                print_time("--- %s seconds ---" % (time.time() - start_time), None)
                for prompt in genai_prompts.prompts:
                    with generator.invoke("[INST] " + prompt + " /[INST]"):
                        pass
                    self.print_debug("finished with prompt", None)
                    print_time("--- %s seconds ---" % (time.time() - start_time), 
                               None)
            print_time("--- %s seconds ---" % (time.time() - start_time), None)
            model_response = self.tokenizer.batch_decode(generator.output)
            
            for i in range(len(genai_prompts.prompts)):
                print("Raw output from Generative AI:")
                print(model_response[i])
                prompts_output.append(model_response[i].split("</s")[0]\
                                      .split("/[INST]")[-1])
                
                
        
        elif self.args.framework == 'transformers':
            inputs = self.tokenizer("[INST] " + genai_prompts.prompts[0] + " /[INST]",
                                    return_tensors="pt").to(self.args.device_map)
            
            outputs = self.model.generate(
                **inputs, max_new_tokens=genai_prompts.max_new_tokens, 
                do_sample=True, temperature=0.1)
            for output in outputs:
                print_time("--- %s seconds ---" % (time.time() - start_time), None)
                model_response = self.tokenizer.decode(output,
                                                       skip_special_tokens=True)
                prompts_output.append(model_response.split("</s")[0]\
                                      .split("/[INST]")[-1])
    
        print_time("--- %s seconds ---" % (time.time() - start_time), None)
        
        return(prompts_output)
    
    def get_text_from_output(self, prompt_output, sep):
        """
        Deconstructs the prompt_output dictionary into its components

        Parameters
        ----------
        prompt_output : dict
            {'prologue' : <str: <output text before the tabular data>>,
             'output_table': <dataframe: <output tabular data>>,
             'epilogue' : <str: <output text after the tabular data>>
             }
        sep : str
            The separator used for the dataframe in output_data.

        Returns
        -------
        table_df : dataframe
            The output dataframe containing the tabular data.
        prologue : str
            The output text before the tabular data.
        output_table : str
            The string in the formate of a .csv file for table_df.
        epilogue : str
            The output text after the tabular data.

        """
        prologue = prompt_output['prologue']
        table_df = prompt_output['output_table']
        output_table = table_df.to_csv(sep=sep, index=False)
        epilogue = prompt_output['epilogue']
        
        return table_df, prologue, output_table, epilogue

    def add_table(self, table_orig, table_df, location, axis):
        """
        Adds the new version's data (table_df) to the base version's table
        (table_orig). This routine is called for adding either row data
        (axis=0) or columnar data (axis=1).

        Parameters
        ----------
        table_orig : VerTable
            The base version's table to be augmented.
        table_df : dataframe
            The new tabular data to be added to the base version to create
            a new table version.
        location : int
            The location in the table (column or row) at which the new 
            version's data should be added to the base version's data.
        axis : 0 (row-based) or 1 (column-based)
            The axis over which the data should be added. Row data will be '0',
            and columnar data will be '1'.

        Returns
        -------
        new_df : dataframe
            The dataframe for the new table version.

        """
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
            
            # if row in new table matches semantic key of old table
            # replace the old row, do not add a new one
            for cur_index, cur_row in old_table.iterrows():
                for new_index, new_row in df.iterrows():
                    matches = True
                    for key in table_orig.semantic_key:
                        if cur_row[key] != new_row[key]:
                            matches = False
                            break
                    if matches:
                        for col in old_table:
                            old_table.at[cur_index, col]\
                                = df.at[new_index, col]
            # some rows of df may not have been in old_table, so we still
            # concatenate the tables together. We finish by dropping
            # duplicates.
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
        new_df = new_df.drop_duplicates()
        self.print_debug(new_df, None)
        if new_df is None:
            print_time(None, "Bad csv format of output")
            time.sleep(10)
            return None
        return new_df
            
    def add_rows_exec(self, table_orig, params):
        """
        Executes the prompts to add rows to the previous version to generate 
        the new version. Note that a large portion of the code is devoted to 
        the 'fake' framework, which is simply a mechanism for testing the code 
        without using the LLM resources.

        Parameters
        ----------
        table_orig : VerTable
            The base table version to which rows will be added.
        params : dict of parameters
            {'num_entries': <int: <number of rows to be added>>,
             'location': <(int: <numeric index of row where the rows will
                                 be added> | 'top' | 'bottom')>}

        Returns
        -------
        genai_prompts.prompts: The list of prompts sent to the LLM (note only
            one prompt is supported at this time)
        rsp: dict
            Dictionary containing the response compnents        
            {'prologue': <str: <text output before the tabular data>>,
             'output_table': <str: <text output of the tabular data>>,
             'epilogue': <str: <text output after the tabular data>>
             }

        """
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
        
        genai_prompts = GenAITablePrompts(self.cache, table_orig, 100000)
        genai_prompts.add_prompt('add_rows', nrows=num_entries)

        print_time(None, None)
        print("Send prompt to Generative AI:")
        print(genai_prompts.prompts[0])
        time.sleep(3)
        
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
            responses.append(f""""{head_nrows}"\n\nThe data is gathered from 
various sources, including but not limited to Encyclopedia Britannica,
National Geographic, and FishBase. The diet, habitat, family, latin name, size,
and common name are directly obtained from these sources. The classification
system (phylum, class, order, suborder, genus, species) is also obtained from
these sources, but sometimes simplified to fit the requested format. The size
is presented as an approximate range for animals that can vary significantly in
size, while for fish, it is a general size indication.""")
            responses.append(
                f"""
{head_nrows}
The second row is generated from the first example you provided, "Shakespeare;Hamlet". I took the liberty of looking up the additional details for this well-known work of classical literature.

The first row is generated from my knowledge of classical literature. Euripides was an ancient Greek playwright, and Medea is one of his surviving tragedies. The characters, themes, and style are based on my understanding of the work.
                """)
            responses.append(
                f"prologue\n\n{head_nrows}\n\nepilogue\n"
                )
            responses.append(
                f"{head_nrows}\nprologue\n\n{head_nrows}\n\nepilogue\n"
                )
            if num_entries >= 2:
                trunc_rows = resp_table.head(2).to_csv(sep=table_orig.format_type[1],
                                                       index=False)
                lines = trunc_rows.split('\n')
                trunc_len = math.floor((len(lines[0]) + len(lines[1]) + len(lines[2]) / 2))
                trunc_rows_rsp = trunc_rows[:trunc_len]
                trunc_rows_rsp_end = trunc_rows[trunc_len:]
                responses.append(
                    f"prologue\n\n{trunc_rows_rsp}\n\nepilogue\n"
                    )
                responses.append(
                    f"prologue\n\n{trunc_rows_rsp}"
                    )
                responses.append(f"{trunc_rows_rsp_end}")
            idx = random.randrange(len(responses)-1)
            # idx = 0 # for testing
            if idx == (len(responses) - 2):
                responses = responses[idx:idx+2]
            else:
                responses = responses[idx:idx+1]
            
        else:
            responses = self.execute_prompts(genai_prompts)

        print_time(None, None)
        print("Received response from Generative AI:")
        print(responses[0])
        time.sleep(3)


        rsp =  self.parse_table_responses(
            table_orig, responses, False, 1, 
            table_orig.table.shape[0] + num_entries) 

        return genai_prompts.prompts, rsp

    def add_rows(self, table_orig, params):
        """
        Executes adding rows to the previous version to generate the new 
        version.

        Parameters
        ----------
        table_orig : VerTable
            The base table version to which rows will be added.
        params : dict of parameters
            {'num_entries': <int: <number of rows to be added>>,
             'location': <(int: <numeric index of row where the rows will
                                 be added> | 'top' | 'bottom')>}

        Returns
        -------
        new_df: dataframe
            Dataframe of the new table version after adding the rows.
        command_dict: dict
            Dictionary containing the add rows response to be put into the
            version's JSON file.
            {'type': <str: <command type (in this case'add_rows')>>,
             'prompt': <str: <text of the prompt given to the LLM>>,
             'start time' : <str: <timestamp of the start of the command
                                   (used for performance monitoring)>>,
             'complete time': <str: <timestamp of the start of the command
                                     (used for performance monitoring)>>,
             'duration (seconds)': <str: <duration required to execute the LLM
                                          command (used for performance 
                                                   monitoring)>>,
             'params': <dict: <parameters for the command (same as input to
                                                           this method)>>,
             'output_table': <str: <textual output of the new tabular data>>,
             'prologue': <str: <textual output of the LLM before the tabular
                                data>>,
             'epilogue': <str: <textual output of the LLM after the tabular
                                data>>,
             'changed': <dict: <a dictionary representation of the rows added>>
             }

        """
        start_time = round(time.time(), 0)
        axis = 0
        prompts_input, prompts_output = \
            self.add_rows_exec(table_orig, params)

        if prompts_output is None or len(prompts_output) == 0:
            print_time(None, "add_rows: Table not found!")
            time.sleep(10)
            return None, None

        table_df, prologue, output_table, epilogue = \
            self.get_text_from_output(prompts_output[0], 
                                      table_orig.format_type[1])
        if output_table == "":
            print_time(None, "add_rows: Table not found!")
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
                        'prologue': prologue,
                        'epilogue': epilogue,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict
            
    def delete_rows(self, table_orig, params):
        """
        Creates a new table version from deleting rows of a base version 

        Parameters
        ----------
        table_orig : VerTable
            The base table version for which rows will be deleted.
        params : dict of parameters
            {'num_entries': <int: <number of rows to be deleted>>,
             'location': <(int: <index of row where the rows will be deleted>>
                           | 'top' | 'bottom')}

        Returns
        -------
        new_df: dataframe
            Dataframe of the new table version after deleting the rows.
        command_dict: dict
            Dictionary containing the delete rows response to be put into the
            version's JSON file. Contains no LLM output as LLM is not 
            necessary.
            {'type': <str: <command type (in this case'del_rows')>>,
             'params': <dict: <parameters for the command (same as input to
                                                           this method)>>,
             'changed': <dict: <a dictionary representation of the rows 
                                deleted>>
             }

        """
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
        self.print_debug(location, None)
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
        Executes the prompts to add columns to the previous version to generate 
        the new version. Note that a large portion of the code is devoted to 
        the 'fake' framework, which is simply a mechanism for testing the code 
        without using the LLM resources.

        Parameters
        ----------
        table_orig : VerTable
            The base table version to which columns will be added.
        params : dict of parameters
            {'num_entries': <int: <number of columns to be added>>,
             'location': <(int: <numeric index of column where the columns will
                                 be added> | 'top' | 'bottom')>}

        Returns
        -------
        genai_prompts.prompts: The list of prompts sent to the LLM (note only
            one prompt is supported at this time)
        rsp: dict
            Dictionary containing the response compnents        
            {'prologue': <str: <text output before the tabular data>>,
             'output_table': <str: <text output of the tabular data>>,
             'epilogue': <str: <text output after the tabular data>>
             }

        """
        ncols = params['num_entries']
        genai_prompts = GenAITablePrompts(self.cache, table_orig, 100000)
        genai_prompts.add_prompt('add_cols', ncols=ncols)

        print_time(None, None)
        print("Send prompt to Generative AI:")
        print(genai_prompts.prompts[0])
        time.sleep(3)
        
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
                        resp_table[colnew] = old_table[col]
                        col_count += 1
                        if col_count >= ncols:
                            break
                if col_count >= ncols:
                    continue
                for col in old_columns:
                    colnew = col + "_NEW"
                    if not colnew in old_columns:
                        resp_table[colnew] = old_table[col]
                        col_count += 1
                        if col_count >= ncols:
                            break
                if col_count >= ncols:
                    continue
                for col in old_columns:
                    resp_table[col + "_NEW"] = old_table[col]
                    col_count += 1
                    if col_count >= ncols:
                        break
                    
                        
            table_str = resp_table.to_csv(sep=table_orig.format_type[1], 
                                          index=False)
            responses.append(
                f"prologue\n\n{table_str}\n\nepilogue\n"
                )
            responses.append(
                f"{table_str}\nprologue\n\n{table_str}\n\nepilogue\n"
                
                )
            responses.append(
                f"prologue\n\n;{table_str}\n\nepilogue\n"
                )
            responses.append(
                f";{table_str}\nprologue\n\n;{table_str}\n\nepilogue\n"
                
                )
            responses.append(
                f";{table_str}\nprologue\n\n{table_str}\n\nepilogue\n"
                
                )
            responses.append(
                f"{table_str}\nprologue\n\n;{table_str}\n\nepilogue\n"
                
                )
            idx = random.randrange(len(responses))
            responses = responses[idx:idx+1]
        else:
                
            responses = self.execute_prompts(genai_prompts)

        print_time(None, None)
        print("Received response from Generative AI:")
        print(responses[0])
        time.sleep(3)

        rsp =  self.parse_table_responses(
            table_orig, responses, True, table_orig.table.shape[0], 
            table_orig.table.shape[0]) 
    
        return genai_prompts.prompts, rsp
    
    def add_cols(self, table_orig, params):
        """
        Executes adding columns to the previous version to generate the new 
        version.

        Parameters
        ----------
        table_orig : VerTable
            The base table version to which columns will be added.
        params : dict of parameters
            {'num_entries': <int: <number of columns to be added>>,
             'location': <(int: <numeric index of column where the columns will
                                 be added> | 'top' | 'bottom')>}

        Returns
        -------
        new_df: dataframe
            Dataframe of the new table version after adding the columns.
        command_dict: dict
            Dictionary containing the add columns response to be put into the
            version's JSON file.
            {'type': <str: <command type (in this case'add_cols')>>,
             'prompt': <str: <text of the prompt given to the LLM>>,
             'start time' : <str: <timestamp of the start of the command
                                   (used for performance monitoring)>>,
             'complete time': <str: <timestamp of the start of the command
                                     (used for performance monitoring)>>,
             'duration (seconds)': <str: <duration required to execute the LLM
                                          command (used for performance 
                                                   monitoring)>>,
             'params': <dict: <parameters for the command (same as input to
                                                           this method)>>,
             'output_table': <str: <textual output of the new tabular data>>,
             'prologue': <str: <textual output of the LLM before the tabular
                                data>>,
             'epilogue': <str: <textual output of the LLM after the tabular
                                data>>,
             'changed': <dict: <a dictionary representation of the columns 
                                added>>
             }

        """
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
        table_df, prologue, output_table, epilogue = \
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
                        'prologue': prologue,
                        'epilogue': epilogue,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict

    def del_cols(self, table_orig, params):
        """
        Creates a new table version from deleting columns of a base version 

        Parameters
        ----------
        table_orig : VerTable
            The base table version for which columns will be deleted.
        params : dict of parameters
            {'num_entries': <int: <number of columns to be deleted>>,
             'location': <(int: <index of column where the columns will be 
                                 deleted>> | 'top' | 'bottom')}

        Returns
        -------
        new_df: dataframe
            Dataframe of the new table version after deleting the columns.
        command_dict: dict
            Dictionary containing the delete columns response to be put into
            the version's JSON file. Contains no LLM output as LLM is not 
            necessary.
            {'type': <str: <command type (in this case'del_cols')>>,
             'params': <dict: <parameters for the command (same as input to
                                                           this method)>>,
             'changed': <dict: <a dictionary representation of the columns 
                                deleted>>
             }

        """
        num_entries = params['num_entries']
        location = params['location']
        indices_elig = []
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
        
        self.print_debug(location, None)
        col_del_list = []
        for i, col in enumerate(df):
            if i in location:
                col_del_list.append(col)
                self.print_debug(None, f"deleting col {col} for location {i}")
                self.print_debug(col_del_list, None)
                
        self.print_debug(location, None)
        self.print_debug(df, None)
        changed_df = pd.DataFrame(columns=col_del_list)
        for i, col in enumerate(df):
            if i in location:
                # col_del_list.append(col)
                self.print_debug(None, f"deleting col {col} for location {i}")
                self.print_debug(changed_df, None)
                
                changed_df[col] = df[col]
        new_df = df.drop(df.columns[location], axis=1)
        
        self.print_debug(new_df, None)
        command_dict = {'type': "del_cols",
                        'params': params,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict

    def fill_na_exec(self, table_orig, params):
        """
        Executes the prompts to fill an N/A value of the previous version to
        generate the new version.
        
        If 'na_loc' is not specified or is None, then it will fill the first
        N/A value in the table. Otherwise, it fills the value at the na_loc
        location and doesn't require that the value replaced actually is N/A.
        (It assumes the caller knows what it is doing.)
        
        Note that a large portion of the code is devoted to the 'fake' 
        framework, which is simply a mechanism for testing the code without 
        using the LLM resources.

        Parameters
        ----------
        table_orig : VerTable
            The base table version to which the specified N/A value will be 
            filled.
        params : dictionary of parameters
            {'na_loc': (<3-tuple: [<int: <numeric index of the row containing
                                          containing the N/A value>>,
                                   <int: <numeric index of the column
                                          containing the N/A value>>,
                                   <str: <column name of the N/A location>>
                                   ] | None)
             }

        Returns
        -------
        genai_prompts.prompts: The list of prompts sent to the LLM (note only
            one prompt is supported at this time)
        rsp: dict
            Dictionary containing the response components        
            {'prologue': <str: <text output before the tabular data>>,
             'output_table': <str: <text output of the tabular data>>,
             'epilogue': <str: <text output after the tabular data>>
             }

        """
    
        na_loc = params['na_loc']
                
        genai_prompts = GenAITablePrompts(self.cache, table_orig, 100000)
        genai_prompts.add_prompt('fill_na', na_loc=na_loc)

        print_time(None, None)
        print("Send prompt to Generative AI:")
        print(genai_prompts.prompts[0])
        time.sleep(3)
        
        
        if self.args.framework == 'fake':
            resp_table = table_orig.table.copy()
            self.print_debug(resp_table, None)
            sem_key = table_orig.semantic_key
            sem_val = []
    
            for col in sem_key:
                sem_val.append(table_orig.table.at[na_loc[1], col])
            num_rows = min(table_orig.table.shape[0], 3)
            num_rows = random.randrange(1, num_rows+1)
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
                    dummy_val = resp_table.at[resp_table.index[1], na_loc[2]]
                else:
                    dummy_val = 0
            else:
                dummy_val = resp_table.at[resp_table.index[0], na_loc[2]]
            resp_table.at[na_loc[1], na_loc[2]] = dummy_val
            small_table = resp_table.iloc[rows,:].to_csv(
                sep=table_orig.format_type[1], index=False)
            self.print_debug(resp_table, None)
            self.print_debug(small_table, None)
            attribute = na_loc[2]
            col_dtype = str(resp_table[attribute].dtype)
            responses = []
            responses.append(
f"""
I have retrieved the missing value for the "{na_loc}" 
attribute for the row with Author=Shakespeare and 
            Title=As You Like It from the website "Performance Database" 
            (<https://www.performingartsdatabase.org/>). The value is 
            "{dummy_val}". Here is the updated table:

{small_table}

Here is the updated table in semi-colon-delimited .csv format:

{small_table}

                """
                )
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
                f"prologue\n\n{small_table}\n\nepilogue\n"
                )
            responses.append(
                f"{small_table}\nprologue\n\n{small_table}\n\nepilogue\n"
                
                )
            no_head_small_table = resp_table.iloc[rows,:].to_csv(
                sep=table_orig.format_type[1], index=False, header=None)
            self.print_debug(no_head_small_table, None)
            responses.append(
                f"prologue\n\n{no_head_small_table}\n\nepilogue\n"
                )
            responses.append(
                f"{no_head_small_table}\nprologue\n\n{no_head_small_table}\n\nepilogue\n"
                )
            responses.append(
                f"{small_table}\nprologue\n\n{no_head_small_table}\n\nepilogue\n"
                )
            small_lines = small_table.split('\n')
            if len(small_lines) > 3:
                small_lines[2] += ";;;;;;;;;;;;"
                small_lines[3] += ";;;;;;;;;;;;"
            x_table = '\n'.join(small_lines)
            responses.append(
                f"{x_table}\nprologue\n\n{x_table}\n\nepilogue\n"
                
                )
            idx = random.randrange(len(responses))
            # idx = 0 # for testing
            responses = responses[idx:idx+1]
            self.print_debug(responses, None)
            time.sleep(3)
            
        else:
                
            responses = self.execute_prompts(genai_prompts)
            
        print_time(None, None)
        print("Received response from Generative AI:")
        print(responses[0])
        time.sleep(3)

        nrows_max = min(table_orig.table.shape[0], 3)
        if nrows_max == 0:
            nrows_max = 1
        rsp =  self.parse_table_responses(table_orig, responses, False, 1, 
                                          nrows_max) 
    
        return genai_prompts.prompts, rsp
    
    def find_all_na(self, df):
        """
        Retrieves the 3-tuple location of all of the N/A values in a dataframe.

        Parameters
        ----------
        df : dataframe
            Dataframe to search for N/A values.

        Returns
        -------
        na_indices : list of 3-tuple of: 
            [<int: <numeric index of the row containing
                    containing the N/A value>>,
             <int: <numeric index of the column
                    containing the N/A value>>,
             <str: <column name of the N/A location>>
             ]
            The list of the N/A locations found.

        """
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
        """
        Creates a new table version from filling in an N/A value of a base 
        version. Note that if the N/A had replaced a previous value (update
        value case), then the new value may or may not be the same (it is at
        the 'whim' of the LLM). If 'na_loc' is None or not present, then the
        first N/A value found in the table will be replaced.

        Parameters
        ----------
        table_orig : VerTable
            The base table version for which columns will be deleted.
        params : dict of parameters
        {'na_loc': (<3-tuple: [<int: <numeric index of the row containing
                                      containing the N/A value>>,
                               <int: <numeric index of the column
                                      containing the N/A value>>,
                               <str: <column name of the N/A location>>
                               ] | None)
         }
        Returns
        -------
        new_df: dataframe
            Dataframe of the new table version after filling in the N/A value.
        command_dict: dict
            Dictionary containing the fill N/A response to be put into the
            version's JSON file.
            {'type': <str: <command type (in this case'fill_na')>>,
             'prompt': <str: <text of the prompt given to the LLM>>,
             'start time' : <str: <timestamp of the start of the command
                                   (used for performance monitoring)>>,
             'complete time': <str: <timestamp of the start of the command
                                     (used for performance monitoring)>>,
             'duration (seconds)': <str: <duration required to execute the LLM
                                          command (used for performance 
                                                   monitoring)>>,
             'params': <dict: <parameters for the command (same as input to
                                                           this method)>>,
             'output_table': <str: <textual output of the new tabular data>>,
             'prologue': <str: <textual output of the LLM before the tabular
                                data>>,
             'epilogue': <str: <textual output of the LLM after the tabular
                                data>>,
             'changed': <dict: <a dictionary representation of the N/A value 
                                filled>>
             }

        """
        start_time = round(time.time(), 0)
        # location = params['location']
        if 'na_loc' not in params or params['na_loc'] is None:
            na_list = self.find_all_na(table_orig.table)
            if len(na_list) == 0:
                print_time(None, "fill_na: no N/A values to retrieve!")
                time.sleep(10)
                return None, None
            if params['location'] == 'random':
                random.shuffle(na_list)
            na_loc = na_list[0]
            params['na_loc'] = na_loc
        else:
            na_loc = params['na_loc']
            
        prompts_input, prompts_output = self.fill_na_exec(table_orig, params)

        if (prompts_input is None or len(prompts_input) == 0 or
            prompts_output is None or len(prompts_output) == 0):
            print_time(None, "fill_na: Table not found!")
            time.sleep(10)
            return None, None
        
        table_df, prologue, output_table, epilogue = \
            self.get_text_from_output(prompts_output[0], 
                                      table_orig.format_type[1])
        if len(output_table) == 0:
            print_time(None, "fill_na: Table not found!")
            time.sleep(10)
            return None, None # we did not find a table in the response, do nothing

        new_df = table_orig.table.copy()
        if na_loc[2] not in table_df:
            print_time(na_loc[2], f"Column {na_loc[2]} not found in response!")
            return None, None
            # new_df.at[na_loc[1], na_loc[2]] = \
            #     table_df.at[table_df.index[0], 
            #                 new_df.columns.get_loc(na_loc[2])]
        else:
            new_df.at[na_loc[1], na_loc[2]] = table_df.at[table_df.index[0], 
                                                          na_loc[2]]
        
        changed_df = pd.DataFrame(columns=[na_loc[2]])
        changed_df.at[na_loc[1], na_loc[2]] = new_df.at[na_loc[1], na_loc[2]]
        
        self.print_debug(changed_df, None)
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
                        'prologue': prologue,
                        'epilogue': epilogue,
                        'changed': changed_df.to_dict()}
        
        return new_df, command_dict
    
    def update_val(self, table_orig, params):
        """
        Creates a new table version from updating a random location of a base 
        version. Note that the new value may or may not be the same (it is at
        the 'whim' of the LLM).

        Parameters
        ----------
        table_orig : VerTable
            The base table version for which columns will be deleted.
        params : dict of parameters: Empty dictionary for now.
            Provided to maintain the same interface as the other commands,
            and in case someone wanted to add some in the future.
        {}

        Returns
        -------
        new_df: dataframe
            Dataframe of the new table version after replacing the location's
            value.
        command_dict: dict
            Dictionary containing the update value response to be put into the
            version's JSON file.
            {'type': <str: <command type (in this case'update_val')>>,
             'prompt': <str: <text of the prompt given to the LLM>>,
             'start time' : <str: <timestamp of the start of the command
                                   (used for performance monitoring)>>,
             'complete time': <str: <timestamp of the start of the command
                                     (used for performance monitoring)>>,
             'duration (seconds)': <str: <duration required to execute the LLM
                                          command (used for performance 
                                                   monitoring)>>,
             'params': <dict: <parameters for the command (same as input to
                                                           this method)>>,
             'output_table': <str: <textual output of the new tabular data>>,
             'prologue': <str: <textual output of the LLM before the tabular
                                data>>,
             'epilogue': <str: <textual output of the LLM after the tabular
                                data>>,
             'changed': <dict: <a dictionary representation of the new 
                                location's value filled>>
             }

        """
        rand_row = random.randrange(table_orig.table.shape[0])
        num_cols_eligible = (table_orig.table.shape[1] 
                             - len(table_orig.semantic_key))
        if num_cols_eligible <= 0:
            return None, None
        rand_col = random.randrange(num_cols_eligible)
        self.print_debug(rand_col, "rand_col")
        i = 0
        use_col = None
        for col in table_orig.table:
            self.print_debug(col, "col")
            if col not in table_orig.semantic_key:
                self.print_debug(i, "i (rand_col)")
                if rand_col == i:
                    use_col = col
                    self.print_debug(use_col, None)
                    break
                i += 1

        if use_col is None:
            return None, None
        
        old_table = table_orig.table.copy()
        row_index = table_orig.table.index[rand_row]
        table_orig.table.at[row_index, use_col] = np.nan
        
        na_loc = (rand_row, row_index, use_col)
        params['na_loc'] = na_loc
        self.print_debug(na_loc, "update_val setting to N/A")
        new_df, command_dict = self.fill_na(table_orig, params)
        table_orig.table = old_table.copy()
        if command_dict is not None:
            command_dict['type'] = 'update_val'
        
        return new_df, command_dict
    
    def new_table(self, table_orig, new_df, command_dict, new_version):
        """
        Generates a new VerTable object (with file) from a base VerTable
        object.

        Parameters
        ----------
        table_orig : VerTable
            Base version VerTable object.
        new_df : dataframe
            Dataframe of new table version.
        command_dict : dict
            Dictionary of metadata for the new table version.
        new_version : int
            New version number.

        Returns
        -------
        None.

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
        
    def parse_header_or_row(self, table_orig, line):
        
        sep = table_orig.format_type[1]
        
        self.print_debug(line, None)
        try:
            df = pd.read_csv(io.StringIO(line), sep=sep)
        except:
            return None, None, None
        raw_field_list = list(df.columns)
        stripped_field_list = [field.strip() for field in raw_field_list]
        matches = 0
        for col1 in table_orig.semantic_key:
            for col2 in stripped_field_list:
                if col1.strip() == col2:
                    matches += 1
                    break
        if matches >= len(table_orig.semantic_key):
            hdr_line = sep.join(stripped_field_list)
            if "Unnamed" in stripped_field_list[0]:
                hdr_line_no_idx = sep.join(stripped_field_list[1:])
            else:
                hdr_line_no_idx = hdr_line
            return hdr_line, hdr_line_no_idx, None
        
        # is it a row without a header? 
        # only if the number of fields matches the original table
        self.print_debug(df.shape[1], None)
        self.print_debug(table_orig.table.shape[1], None)
        if df.shape[1] == table_orig.table.shape[1]:
            self.print_debug(stripped_field_list, None)
            return None, None, sep.join(stripped_field_list)
        
        return None, None, None

    def parse_table(self, table_orig, lines):
        
        if len(lines) == 0:
            return None, False
        
        sep = table_orig.format_type[1]

        # Sometimes the Gen AI encapsulates the entire table within 
        # double-quotes        
        for i in range(len(lines)):
            lsplit = lines[i].split('"')
            if len(lsplit) == 2:
                lines[i] = ''.join(lsplit)
                
        
        hdr_line, hdr_line_no_idx, row_line = \
            self.parse_header_or_row(table_orig, lines[0])
        self.print_debug(hdr_line, "hdr_line")
        self.print_debug(hdr_line_no_idx, "hdr_line_no_idx")
        self.print_debug(row_line, "row_line")
            
        self.print_debug(row_line, "row_line")

        if hdr_line is None and hdr_line_no_idx is None and row_line is None:
            return None, False
        
        header_list = list(table_orig.table.columns)
        self.print_debug(header_list, "header_list")
        
        our_df = None
        if len(lines) == 1 or len(lines[1]) == 0:
            try:
                if hdr_line is not None:
                    return pd.read_csv(io.StringIO(hdr_line), sep=sep), True
                elif hdr_line_no_idx is not None:
                    return pd.read_csv(io.StringIO(hdr_line_no_idx), sep=sep), True
                else: # row_line is not None
                    return pd.read_csv(io.StringIO(row_line), sep=sep,
                                       header=None, names=header_list), False
            except:
                return None, False

        try:
            if hdr_line is not None:
                dfi = pd.read_csv(io.StringIO(hdr_line), sep=sep)
                our_df = dfi
                self.print_debug(dfi, None)
                self.print_debug(dfi.shape, None)
            if hdr_line_no_idx is not None:
                dfh = pd.read_csv(io.StringIO(hdr_line_no_idx), sep=sep)
                our_df = dfh
                self.print_debug(dfh, None)
                self.print_debug(dfh.shape, None)
            if row_line is not None:
                dfr = pd.read_csv(io.StringIO(row_line), sep=sep, header=None,
                                  names=header_list)
                our_df = dfr
            df1 = pd.read_csv(io.StringIO(lines[1]), sep=sep)
        except:
            return None, False
        self.print_debug(df1, None)
        self.print_debug(df1.shape, None)
        

        if row_line is not None:
            # we are a headerless table
            for num_lines in range(2, len(lines)):
                try:
                    # dfl = pd.read_csv(io.StringIO(lines[num_lines-1]), 
                    #                   header=None, sep=sep,
                    #                   names=header_list)
                    dfl = pd.read_csv(io.StringIO(lines[num_lines-1]), sep=sep)
                    self.print_debug(dfl, None)
                    if dfl is None:
                        return our_df, False
                    self.print_debug(dfl.shape, None)
                    self.print_debug(dfr.shape, None)
                    if dfl.shape[1] != dfr.shape[1]:
                        return our_df, False
                except:
                    return our_df, False
                
                table_lines = [row_line]
                table_lines.extend(lines[1:num_lines])
                table_text = "\n".join(table_lines)
                try:
                    df = pd.read_csv(io.StringIO(table_text), sep=sep,
                                     header=None,
                                     names=header_list)
                    self.print_debug(df, None)
                    if df is not None:
                        our_df = df
                except:
                    return our_df, False
            return our_df, False
        
        self.print_debug(df1.shape, None)
        self.print_debug(dfh.shape, None)
        self.print_debug(hdr_line_no_idx, None)
        if hdr_line_no_idx is not None and df1.shape[1] == dfh.shape[1]:
            self.print_debug(None, "Shapes are equal")
            # we match the header without the index
            for num_lines in range(2, len(lines)):
                try:
                    dfl = pd.read_csv(io.StringIO(lines[num_lines-1]), sep=sep)
                    self.print_debug(dfl, None)
                    if dfl is None:
                        return our_df, True
                    self.print_debug(dfl.shape, None)
                    self.print_debug(dfh.shape, None)
                    if dfl.shape[1] != dfh.shape[1]:
                        return our_df, True
                except:
                    return our_df, True
                
                table_lines = [hdr_line_no_idx]
                table_lines.extend(lines[1:num_lines]) # should be not +1
                table_text = "\n".join(table_lines)
                try:
                    df = pd.read_csv(io.StringIO(table_text), sep=sep)
                    self.print_debug(df, None)
                    if df is not None:
                        our_df = df
                except:
                    return our_df, True
            return our_df, True

        if hdr_line is not None and df1.shape[1] == dfi.shape[1]:
            # we match the header with the index
            for num_lines in range(2, len(lines)):
                try:
                    dfl = pd.read_csv(io.StringIO(lines[num_lines-1]), sep=sep)
                    self.print_debug(df, "first df")
                    if dfl is None:
                        return our_df, True
                    self.print_debug(dfl.shape, None)
                    self.print_debug(dfi.shape, None)
                    if dfl.shape[1] != dfi.shape[1]:
                        return our_df, True
                except:
                    return our_df, True
                
                table_lines = [hdr_line]
                table_lines.extend(lines[1:num_lines])
                table_text = "\n".join(table_lines)
                try:
                    df = pd.read_csv(io.StringIO(table_text), sep=sep)
                    self.print_debug(df, "second df")
                    self.print_debug(df, None)
                    if df is not None:
                        our_df = df
                except:
                    return our_df, True
            return our_df, True

        return our_df, True
                

    def find_valid_csv_tables(self, table_orig, text):
 
        valid_tables = []
        lines = text.split("\n")
        line_num = 0
        
        while line_num < len(lines):
            df, has_header = self.parse_table(table_orig, lines[line_num:])
            if df is not None:
                if line_num > 0:
                    prologue_text = "\n".join(lines[:line_num-1])
                else:
                    prologue_text = ""
                line_num = line_num + df.shape[0] + 1
                if line_num == (len(lines) - 1):
                    epilogue_text = ""
                else:
                    epilogue_text = "\n".join(lines[line_num+1:])
                    
                csv_table = (prologue_text, has_header, df, epilogue_text)
                valid_tables.append(csv_table)
            else:
                line_num += 1
                
        return valid_tables
            
            # csv_table[0] is prologue_text
            # csv_table[1] is True/False for has header
            # csv_table[2] is table dataframe
            # csv_table[3] is epilogue text
    
    def parse_table_responses(self, table, responses, header_required, 
                              nrows_min, nrows_max):
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
        self.print_debug(nrows_min, None)
        self.print_debug(nrows_max, None)
        rsp = ""
        for response in responses:
            rsp = rsp + response

        # Note: indent the following if you're really executing multiple
        # prompts            
        self.print_debug(rsp, None)

        valid_csv_tables = self.find_valid_csv_tables(table, rsp)
        self.print_debug(valid_csv_tables, None)

        # loop through valid tables
        col_valid_csv_table = None
        for csv_table in valid_csv_tables:
            # csv_table[0] is prologue_text
            # csv_table[1] is True/False for has header
            # csv_table[2] is table dataframe
            # csv_table[3] is epilogue text
            if header_required and not csv_table[1]:
                continue
            nrows = csv_table[2].shape[0]
            if nrows < nrows_min or nrows > nrows_max:
                continue
            col_valid_csv_table = csv_table # look for the final table
        self.print_debug(col_valid_csv_table == None, None)
        # time.sleep(30)
            
        if col_valid_csv_table is not None:
            table_response = {
                'prologue': col_valid_csv_table[0],
                'output_table': col_valid_csv_table[2].copy(),
                'epilogue': col_valid_csv_table[3]
                }
            self.print_debug(table_response, None)
            table_responses.append(table_response)
                
        return(table_responses)
    
    def command_exec(self, command):
        # start_time = time.time()
        
        cache = self.cache
        table_name = self.args.name
        
        new_version = cache.get_missing_ver_for_table(table_name)
        self.print_debug(new_version, None)
        if self.args.lineage == "random":
            table_orig = cache.get_prev_table_random_from_cache(table_name,
                                                                new_version)
        else: # "sequence"
            table_orig = cache.get(table_name, new_version - 1)
        assert table_orig is not None
        self.print_debug(table_orig.version, None)
        self.print_debug(table_orig.semantic_key, None)
        assert(len(table_orig.semantic_key) > 0)
        if table_orig.table is None:
            was_none = True
            table_orig.read()
        else:
            was_none = False
        self.print_debug(table_orig.semantic_key, None)
        command_type = command['type']
        params = command['params']
        params_list = get_params_list(params)
        random.shuffle(params_list)
        params = params_list[0]
        print_time(params, (f"Starting operation...{command_type} for "
                            + f"{table_name}, base version "
                            + f"{table_orig.version}, new version "
                            + f"{new_version}"))
        
        if command_type == "add_rows":
            new_df, command_dict = self.add_rows(table_orig, params)
        elif command_type == "del_rows":
            new_df, command_dict = self.delete_rows(table_orig, params)
        elif command_type == "add_cols":
            new_df, command_dict = self.add_cols(table_orig, params)
        elif command_type == "del_cols":
            new_df, command_dict = self.del_cols(table_orig, params)
        # elif command_type == "fill_na":
        #     new_df, command_dict = self.fill_na(table_orig, params)
        elif command_type == "update_val":
            new_df, command_dict = self.update_val(table_orig, params)
    
        if new_df is None:
            return False
        
        self.new_table(table_orig, new_df, command_dict, new_version)

        if was_none:
            table_orig.purge()

        print_time(command_type, "Finished successfully")
        time.sleep(3)
        return True
        
    def main(self, args):
        """
        Main program to create a series of table versions.

        Parameters
        ----------
        args : args
            Command-line arguments.

        $ python gbv_main.py -h
        2024-05-03 18:34:57.651339
        Starting globally...
        
        usage: gbv_main.py [-h] [--debug] [-s ORIG_DELIM] [-l LINEAGE] [-g DEVICE_MAP]
                           [-n NUM_VER] [-i MAX_ITER] [-f FRAMEWORK] [-d DTYPE]
                           [-c CMD] [-a MAX_ATTEMPTS]
                           name tablesdir
        
        Auto-generates versions of base file using generative AI
        
        positional arguments:
          name                  Filename of the table without extension (only .csv
                                extension is supported)
          tablesdir             Directory locaton of the tables. Default is the
                                current working directory.
        
        options:
          -h, --help            show this help message and exit
          --debug               Turns on debug logging to stdout. Default is off.
          -s ORIG_DELIM, --sep ORIG_DELIM
                                Column separator of the original file. Only used
                                before versions have been generated. Default is comma
          -l LINEAGE, --lineage LINEAGE
                                Type of lineage for versions created: "sequence"
                                (default) | "random"
          -g DEVICE_MAP, --gpu DEVICE_MAP
                                Specify type of GPU: "cuda" (default) | "mps". Default
                                is "auto"
          -n NUM_VER, --numver NUM_VER
                                Number of versions to create. Default is 10.
                                Unsuccessful attempts do not count
          -i MAX_ITER, --maxiter MAX_ITER
                                Maximum number of table modification attempts. Default
                                is 20.
          -f FRAMEWORK, --framework FRAMEWORK
                                Model framework type: nnsight | transformers | fake.
                                Default is nnsight
          -d DTYPE, --datatype DTYPE
                                Data type for model. Use "float16" to reduce
                                footprint. Default is None
          -c CMD, --command CMD
                                Specific command to run exlusively. Default is None,
                                which means to use a hard-coded script.
          -a MAX_ATTEMPTS, --maxattempts MAX_ATTEMPTS
                                Maximum number of attempts for each command. Default
                                is 3.        Returns
        -------
        None.

        """
        
        self.args = args
        
        numver = 0
        command_idx = None
        num_attempts = 0
        cmd_plan_pos = 0
        for idx in range(self.args.max_iter):
            if self.args.cmd is not None:
                for i, cmd in enumerate(COMMANDS):
                    if self.args.cmd == cmd['type']:
                        command_idx = i
                        break
            print_time(command_idx, "before check")
            if command_idx is None:
                if cmd_plan_pos < len(CMD_PLAN):
                    command_idx = CMD_PLAN[cmd_plan_pos]
                    print_time(command_idx, "plan")
                else:
                    command_idx = random.randrange(len(COMMANDS))
                    print_time(command_idx, "random")
                    
            command = COMMANDS[command_idx]
            if self.command_exec(command):
                numver += 1
                cmd_plan_pos += 1
                command_idx = None
            else:
                num_attempts += 1
                print_time(None, "command failed")
                time.sleep(10)
                if num_attempts >= self.args.max_attempts:
                    cmd_plan_pos += 1
                    command_idx = None
                elif idx < (self.args.max_iter - 1):
                    print_time(None, "attempting again")
                    time.sleep(3)
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
        '--debug', dest='debug', action='store_true', default=False,
        help='Turns on debug logging to stdout. Default is off.')
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
        help='Maximum number of table modification attempts. Default is 20.')
    parser.add_argument(
        '-f', '--framework', dest='framework', type=str, default='nnsight',
        help=('Model framework type: nnsight | transformers | fake. '
              + 'Default is nnsight'))
    parser.add_argument(
        '-d', '--datatype', dest='dtype', type=str, default=None,
        help=('Data type for model. Use "float16" to reduce footprint. '
              + ' Default is None'))
    parser.add_argument(
        '-c', '--command', dest='cmd', type=str, default=None,
        help=('Specific command to run exlusively. Default is None, which '
              + 'means to use a hard-coded script.'))
    parser.add_argument(
        '-a', '--maxattempts', dest='max_attempts', type=int, default=3,
        help='Maximum number of attempts for each command. Default is 3.')
    parser.add_argument('name', type=str,
        help=('Filename of the table without extension '
              + '(only .csv extension is supported)'))
    parser.add_argument('tablesdir', type=str, default=".",
        help=('Directory locaton of the tables. ' 
              + 'Default is the current working directory.'))
    
    args = parser.parse_args()
    
    print_time(args, None)
    
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