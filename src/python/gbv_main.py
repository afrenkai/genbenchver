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
import json
import itertools
import inspect

# Note only semi-colon-delimited csv files are supported at this time

DEBUG = True

FAKE_MODEL = False

LINEAGE = "sequence"
# LINEAGE = "random"

ENVIRONMENT = "turing.wpi.edu"
# ENVIRONMENT = "local_macos"

TABLES_VERSION_DELIMITER = "_"

# MODEL_SPEC = 'learnanything/llama-7b-huggingface'
MODEL_SPEC = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# MODEL_SPEC = 'mistralai/Mistral-7B-v0.1'

FORMAT_CSV_DELIMITER = ';'
FORMAT_CSV_DELIMITER_NAME = "semi-colon"
FORMAT_CSV = '.csv'
FORMAT_TYPE = (FORMAT_CSV, FORMAT_CSV_DELIMITER, FORMAT_CSV_DELIMITER_NAME)

if ENVIRONMENT == "turing.wpi.edu":
    MODEL_TYPE = 'nnsight'
    DEVICE_MAP = "cuda"
    TABLES_FOLDER = "tables3"
else: # ENVIRONMENT == "local_macos"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    MODEL_TYPE = 'transformers'
    DEVICE_MAP = "mps"
    TABLES_FOLDER = "tables3"

COMMANDS = [
    {'type': 'add_row',
     'params': {'num_entries' : [1, 2, 3, 4, 5],
                'location' : ['top', 'bottom', 'random']}},
    {'type': 'del_row',
     'params': {'num_entries' : [1, 2, 3, 4, 5],
                'location' : ['random']}},
    {'type': 'add_col',
     'params': {'num_entries' : [1, 2, 3], # [1, 2, 3, 4, 5],
                'location' : ['left', 'right', 'random']}},
    {'type': 'del_col',
     'params': {'num_entries' : [1, 2, 3, 4, 5],
                'location' : ['random']}},
    {'type': 'fill_na',
    'params': {'location' : ['first', 'random']}}
    ]

CMD_PLAN = [
    4, # fill_na
    4, # fill_na
    0, # add_row
    2, # add_col
    3, # del_col
    1, # del_row
    4, # fill_na
    2, # add_col
    1, # del_row
    0, # add_row
    ]

TIME_START = time.time()
DATETIME_START = dt.datetime.now()

random.seed(TIME_START)

if MODEL_TYPE == 'transformers':
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    import torch
elif MODEL_TYPE == 'nnsight':
    from nnsight import LanguageModel
    from transformers import AutoTokenizer

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

def print_time_force(var, text):
    fmt_text = f"\n{dt.datetime.now()}, {time.time() - TIME_START} seconds\n"
    if var is not None:
        varstr = get_variable_name(var)
        if varstr is not None and varstr != "var":
            fmt_text += f"{get_variable_name(var)} =\n{var}\n"
        else:
            fmt_text += f"\n{var}\n"
    if text is not None:
        fmt_text += f"{text}\n"
    print(fmt_text, flush=True)

def print_time(var, text):
    if not DEBUG:
        return
    print_time_force(var, text)

print_time(None, "Starting globally...")

# def find_all_na(df):
#     na_indices = []
#     i = 0
#     for index, row in df.iterrows():
#         for col in df:
#             if pd.isnull(df.at[index, col]):
#                 na_indices.append((i, index, col))
#         i += 1
#     return na_indices
    
def find_all_na(df):
    na_df = df.isna().copy()
    print_time(na_df, None)
    na_indices = []
    i = 0
    for index, row in na_df.iterrows():
        for col in na_df:
            if na_df.at[index, col]:
                na_indices.append((i, index, col))
        i += 1
    return na_indices
    
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

def update_ver_table_cache(vtindex, name, table, version):
    """
    

    Parameters
    ----------
    vtindex : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.
    version : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if name not in vtindex:
        vtindex[name] = {}
    name_dict = vtindex[name]
    name_dict[version] = {'table': table}
    
def build_ver_table_cache(folder, version_delimiter):
    """
    

    Parameters
    ----------
    folder : TYPE
        DESCRIPTION.
    version_delimiter : TYPE
        DESCRIPTION.

    Returns
    -------
    cache : TYPE
        DESCRIPTION.

    """
    cache = {}
    # get base files first
    for filename in os.listdir(folder):
        # for now, only csv supported
        if filename.endswith(".csv"):
            fullname = filename.split(".")[0]
            s = fullname.split(version_delimiter)
            description = None
            lineage = []
            semantic_key = ""
            preamble = None
            postamble = None
            command_dict = None
            if len(s) == 1:
                name = s[0]
                version = 0
                json_fn = name + ".json"
            elif len(s) >= 2 and s[-1].isdecimal():
                name = version_delimiter.join(s[0:-1])
                version = int(s[-1])
                json_fn = name + version_delimiter + str(version) + ".json"
            else:
                name = version_delimiter.join(s)
                version = 0
                json_fn = name + ".json"
            json_ffn = os.path.join(folder, json_fn)
            if os.path.exists(json_ffn):
                with open(json_ffn) as fp:
                    json_dict = json.load(fp)
                if json_dict is not None:
                    description = json_dict['description']
                    if 'lineage' in json_dict:
                        lineage = json_dict['lineage']
                    if 'semantic_key' not in json_dict:
                        if FAKE_MODEL:
                            semantic_key = ["Model", "Make", "Year"]
                    else:
                        semantic_key = json_dict['semantic_key']
                    if 'preamble' in json_dict:
                        preamble = json_dict['preamble']
                    if 'postamble' in json_dict:
                        postamble = json_dict['postamble']
                    if 'command' in json_dict:
                        command_dict = json_dict['command']
            print_time(semantic_key, None)
            print_time(lineage, None)
            table = VerTable(None, folder, name, description, preamble, 
                             postamble, semantic_key, version_delimiter, 
                             FORMAT_TYPE, version, lineage, command_dict)
            if table is not None:
                update_ver_table_cache(cache, name, table, version)
    return cache

def set_node_ver_table_cache(vtindex, table):
    """
    

    Parameters
    ----------
    vtindex : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    update_ver_table_cache(vtindex, table.name, table, table.version)
                
def unset_node_ver_table_cache(vtindex, table):
    """
    

    Parameters
    ----------
    vtindex : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    update_ver_table_cache(vtindex, table.name, None, table.version)

def get_high_ver_for_table(cache, name):
    """
    

    Parameters
    ----------
    cache : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    new_version : TYPE
        DESCRIPTION.

    """
    return max(list(cache[name].keys())) 

def get_next_ver_for_table(cache, name):
    """
    

    Parameters
    ----------
    cache : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    new_version : TYPE
        DESCRIPTION.

    """
    return get_high_ver_for_table(cache, name) + 1

def get_table_random_from_cache(cache, name):
    """
    

    Parameters
    ----------
    cache : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    table : TYPE
        DESCRIPTION.

    """
    if name in cache:
        r = random.randrange(len(cache[name]))
        for i, version in enumerate(cache[name]):
            if i == r:
                table = cache[name][version]['table']
                if table is not None:
                    return table
    return None
                            

def get_table_from_cache(cache, name, version):
    """
    

    Parameters
    ----------
    cache : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    version : TYPE
        DESCRIPTION.

    Returns
    -------
    table : TYPE
        DESCRIPTION.

    """
    if name in cache:
        if version in cache[name] and 'table' in cache[name][version]:
            table = cache[name][version]['table']
            if table is not None:
                return table
    return None

def read_table_from_cache(cache, name, version):
    """
    

    Parameters
    ----------
    cache : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    version : TYPE
        DESCRIPTION.

    Returns
    -------
    table : TYPE
        DESCRIPTION.

    """
    table = get_table_from_cache(cache, name, version)
    if table is not None:
        table.read()
    return table

def add_table_to_cache(vtindex, table):
    """
    

    Parameters
    ----------
    vtindex : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if table.name not in vtindex:
        vtindex[table.name] = {}
    cur_ver_dict = vtindex[table.name]
    if table.version not in cur_ver_dict:
        cur_ver_dict[table.version] = {}
    
    if 'table' in cur_ver_dict[table.version]:
        print_time_force(None, "table already set in cache! Overwriting...")
    cur_ver_dict[table.version]['table'] = table

class VerTable:
    """
    
    """
    
    def __init__(self, table, folder, name, description, preamble, postamble, 
                 semantic_key, version_delimiter, format_type, version, 
                 lineage, command_dict):
        """
        

        Parameters
        ----------
        table : TYPE
            DESCRIPTION.
        folder : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        description : TYPE
            DESCRIPTION.
        preamble : TYPE
            DESCRIPTION.
        postamble : TYPE
            DESCRIPTION.
        semantic_key : TYPE
            DESCRIPTION.
        format_type : TYPE
            DESCRIPTION.
        version : TYPE
            DESCRIPTION.
        lineage : TYPE
            DESCRIPTION.
        command_dict : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.table = table
        self.folder = folder
        self.name = name
        self.description = description
        self.preamble = preamble
        self.postamble = postamble
        self.format_type = format_type
        print_time(semantic_key, None)
        self.semantic_key = semantic_key
        self.version = version
        if self.version == 0:
            self.version_str = ""
        else:
            self.version_str = "_" + str(self.version)
        if lineage is None:
            lineage = []
        print_time(lineage, "C")
        self.lineage = lineage
        self.command_dict = command_dict
        print_time(self.name, None)
        print_time(self.version_str, None)
        self.filespec = os.path.join(self.folder, self.name + self.version_str)
        self.input_table_entries = 0
        self.input_table_attributes = 0
        
        json_dict = None
        if self.description is None:
            self.description = ""
            if os.path.exists(self.filespec + ".json"):
                with open(self.filespec + ".json", "r") as fp:
                    json_dict = json.load(fp)
                    print("")
                    print("json_dict")
                    print_time(json_dict, None)
                    self.description = json_dict['description']
                    if 'lineage' in json_dict:
                        self.lineage = json_dict['lineage']
                    if FAKE_MODEL:
                        if 'semantic_key' not in json_dict:
                            self.semantic_key = ["Model, Make, Year"]
                    else:
                        self.semantic_key = json_dict['semantic_key']
                    if 'preamble' in json_dict:
                        self.preamble = json_dict['preamble']
                    if 'postamble' in json_dict:
                        self.postamble = json_dict['postamble']
                    if 'command' in json_dict:
                        self.command_dict = json_dict['command']
            
        print_time(lineage, "D")
        if json_dict is None:
            json_dict = {}
            json_dict['description'] = self.description
            json_dict['lineage'] = self.lineage
            json_dict['semantic_key'] = self.semantic_key
            print_time(self.semantic_key, None)
            json_dict['preamble'] = self.preamble
            json_dict['postamble'] = self.postamble
            json_dict['command'] = self.command_dict
            with open(self.filespec + ".json", 'w') as fp:
                json.dump(json_dict, fp, indent=4)
            
        # if self.table is None:
        #     self.read()
        if type(self.table) == str:
            self.write()
        elif self.table is not None: # type is dataframe
            self.write()

        if self.table is not None:
            self.num_entries = self.table.shape[0]
            self.num_table_attributes = self.table.shape[1]
            
    def __str__(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.table is not None:
            # print_time("returning csv")
            # time.sleep(10)
            return self.table.to_csv(sep=self.format_type[1], index=False)
        return "None"
    
    def write(self):
        """
        

        Returns
        -------
        None.

        """
        # for now, csv only format type supported
        filename = self.filespec + self.format_type[0]
        if type(self.table) == str:
            with open(filename, "w") as fp:
                fp.write(self.table)
            self.table = pd.read_csv(filename, sep=self.format_type[1])
        self.table.reset_index(drop=True, inplace=True)
        self.table.to_csv(filename, sep=self.format_type[1], index=False)
            
    def read(self):
        """
        

        Returns
        -------
        None.

        """
        if self.table is None:
            filename = self.filespec + self.format_type[0]
            self.table = pd.read_csv(filename, sep=self.format_type[1])
            self.num_entries = self.table.shape[0]
            self.num_table_attributes = self.table.shape[1]
            self.table.reset_index(drop=True, inplace=True)
            self.table.to_csv(filename, sep=self.format_type[1], index=False)
            return True
        return False
    
    def purge(self):
        self.table = None
        
    def get_num_entries(self):
        was_none = self.read()
        num_entries = self.table.shape[0]
        if was_none:
            self.purge()
        return num_entries
    
    def get_num_attributes(self):
        was_none = self.read()
        num_entries = self.table.shape[1]
        if was_none:
            self.purge()
        return num_entries
    
    def update(self, table):
        """
        

        Parameters
        ----------
        table : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.table = table
        self.write()
        
    def get_table_header_only(self):
        """
        

        Returns
        -------
        empty_table : TYPE
            DESCRIPTION.

        """
        if self.table is None:
            filename = self.filespec + self.format_type[0]
            empty_table = pd.read_csv(filename, sep=self.format_type[1],
                                      index_col=0, nrows=0)
        else:
            empty_table = self.table.drop(self.table.index) 
        return empty_table
    
    def get_ineligible_columns_header_only(self):
        """
        

        Returns
        -------
        empty_table : TYPE
            DESCRIPTION.

        """
        if self.table is None:
            filename = self.filespec + self.format_type[0]
            empty_table = pd.read_csv(filename, sep=self.format_type[1],
                                      index_col=0, nrows=0)
        else:
            empty_table = self.table.drop(self.table.index)
        for i in self.lineage:
            table = get_table_from_cache(cache, self.name, i)
            if table.command_dict['type'] == 'del_col':
                for col in table.command_dict['changed']:
                    empty_table[col] = None
        return empty_table
    
    def get_semantic_key(self):
        return self.semantic_key
    
    def get_description(self):
        return self.description
    
    def get_table_key_only(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.read()
        return self.table[self.semantic_key]
    
    def get_ineligible_rows_key_only(self, cache):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.read()
        df = self.table.copy()
        if len(self.semantic_key) == 0:
            dfsk = df.copy()
        else:
            dfsk = df[self.semantic_key].copy()
        dfsk = dfsk.drop_duplicates()
        for i in self.lineage:
            table = get_table_from_cache(cache, self.name, i)
            if table.command_dict['type'] == 'del_row':
                extend_df = pd.DataFrame(table.command_dict['changed'])
                if len(self.semantic_key) == 0:
                    extend_dfsk = extend_df.copy()
                else:
                    extend_dfsk = extend_df[self.semantic_key].copy()
                extend_dfsk = extend_dfsk.drop_duplicates()
                print_time(extend_dfsk, None)
                print_time(dfsk, None)
                dfsk = pd.concat([dfsk, extend_dfsk])
                dfsk = dfsk.drop_duplicates()
        # dfsk = df[self.semantic_key].copy()
        # dfsk = dfsk.drop_duplicates()
        return dfsk
    
def new_table(cache, orig_table, version_delimiter, preamble, new_df, postamble, 
              command_dict, new_version):
    """
    

    Parameters
    ----------
    cache : TYPE
        DESCRIPTION.
    orig_table : TYPE
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
    lineage = orig_table.lineage
    print_time(lineage, "E")
    if lineage is None:
        lineage = []
    if orig_table.version > 0:
        if len(lineage) > 0:
            assert(lineage[-1] != orig_table.version)
        lineage.append(orig_table.version)
    print_time(lineage, "A")
    new_table = VerTable(new_df, orig_table.folder, orig_table.name,
                         orig_table.description, preamble, postamble,
                         orig_table.semantic_key, version_delimiter, 
                         orig_table.format_type,
                         new_version, lineage, command_dict)
    add_table_to_cache(cache, new_table)

def add_table(orig_table, data, location, axis):
    """
    

    Parameters
    ----------
    orig_table : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    location : TYPE
        DESCRIPTION.
    axis : TYPE
        DESCRIPTION.

    Returns
    -------
    new_df : TYPE
        DESCRIPTION.

    """
    if type(data) == str:
        try:
            df = pd.read_csv(io.StringIO(data), sep=orig_table.format_type[1])
            df.reset_index(drop=True, inplace=True)
            # lineterminator='\n'
        except:
            return None
    else: # dataframe
        df = data.copy()
        df.reset_index(drop=True, inplace=True)
    # if axis == 0:
        
        # nrows = orig_table.table.shape[axis]
        # if location == 'top':
        #     new_df = pd.concat([df, orig_table.table], axis=axis)
        #     split_loc = 0
        # elif location == 'bottom':
        #     new_df = pd.concat([orig_table.table, df], axis=axis)
        #     split_loc = (nrows-1)
        # else: # location == 'random'
        #     # randomly split the table
            # split_loc = random.randrange(nrows)
    #         if split_loc == 0:
    #             new_df = pd.concat([df, orig_table.table], axis=axis)
    #         elif split_loc == (nrows-1):
    #             new_df = pd.concat([orig_table.table, df], axis=axis)
    #         else:
    #             top = orig_table.table.head(split_loc)
    #             bottom = orig_table.table.tail(nrows - split_loc)
    #             new_df = pd.concat([top, df, bottom], axis=axis)
    # else:
    #     ncols = orig_table.table.shape[axis]
    #     if location == 'left':
    #         new_df = pd.concat([df, orig_table.table], axis=axis)
    #         split_loc = 0
    #     elif location == 'right':
    #         new_df = pd.concat([orig_table.table, df], axis=axis)
    #         split_loc = (ncols-1)
    #     else: # location == 'random'
    #         # randomly split the table
    #         split_loc = random.randrange(ncols)
    #         if split_loc == 0:
    #             new_df = pd.concat([df, orig_table.table], axis=axis)
    #         elif split_loc == (ncols-1):
    #             new_df = pd.concat([orig_table.table, df], axis=axis)
    #         else:
    #             left = orig_table.table.iloc[:, 0:split_loc]
    #             right = orig_table.table.iloc[:, split_loc:ncols]
    #             new_df = pd.concat([left, df, right])
    was_none = False
    if orig_table.table is None:
        orig_table.read()
        was_none = True
    n_entries = orig_table.table.shape[axis]
    n_rows = orig_table.table.shape[0]
    n_cols = orig_table.table.shape[1]
    old_table = orig_table.table.copy()
    if axis == 0:
        for col in df:
            if col not in old_table:
                old_table[col] = np.nan
        for col in old_table:
            if col not in df:
                df[col] = np.nan
        print_time(old_table, "row")
        print_time(df, "row")
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
        
        print_time(df, "col")
        print_time(old_table, "col")
        new_df = old_table.merge(df, how='outer', on=orig_table.semantic_key)
        print_time(new_df, "col")
        
        cols_new = []
        if location == 0:
            
            for col in df:
                if col not in orig_table.semantic_key:
                    cols_new.append(col)
            for col in old_table:
                cols_new.append(col)

            # print("")
            # print("df")
            # print_time(df)
            # print("")
            # print("old_table")
            # print_time(old_table)
            # new_df = pd.concat([df, old_table], axis=axis)
            # print("")
            # print("new_df")
            # print_time(new_df)
        elif location == (n_entries-1):
            
            for i, col in enumerate(old_table):
                if i < location:
                    cols_new.append(col)
            for col in df:
                if col not in orig_table.semantic_key:
                    cols_new.append(col)
            for i, col in enumerate(old_table):
                if i >= location:
                    cols_new.append(col)
                    
            # print("")
            # print("df")
            # print_time(df)
            # print("")
            # print("old_table")
            # print_time(old_table)
            # new_df = pd.concat([old_table, df], axis=axis)
            # print("")
            # print("new_df")
            # print_time(new_df)
        else:
            
            for col in old_table:
                cols_new.append(col)
            for col in df:
                if col not in orig_table.semantic_key:
                    cols_new.append(col)
                
            # print("")
            # print("location, n_entries, n_rows")
            # print_time(str(location) + " " + str(n_entries) + " " 
            #            + str(n_rows))
            # before = old_table.iloc[:, :location].copy()
            # print("")
            # print("before")
            # print_time(before)
            # after = old_table.iloc[:, location:].copy()
            # print("")
            # print("after")
            # print_time(after)
            # new_df = pd.concat([before, df, after], axis=axis)
            # print("")
            # print("new_df")
            # print_time(new_df)
        new_df = new_df[cols_new]
        
    new_df.reset_index(drop=True, inplace=True)            
    print_time(new_df, None)
    if was_none:
        orig_table.table = None
    return new_df
        
def build_model(model_type, model_id):
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
    if FAKE_MODEL:
        return None, None
    if model_type == 'nnsight':
        tokenizer = AutoTokenizer.from_pretrained(model_id, unk_token="<unk>",
                                                  pad_token='[PAD]')
        model = LanguageModel(model_id, device_map='auto', tokenizer=tokenizer)
        
    elif model_type == 'transformers':
        # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        if ENVIRONMENT == "turing.wpi.edu":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map='auto') #, torch_dtype=torch.float16)
        else: # ENVIRONMENT == "local_macos"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map='auto', torch_dtype=torch.float16)
        # model = torch.nn.DataParallel(model, device_ids=[0,1])
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, unk_token="<unk>")
    return model, tokenizer

def execute_prompts(model_type, tokenizer, model, max_new_tokens, prompts):
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
    
    if model_type == 'nnsight':
        with model.generate(max_new_tokens=max_new_tokens, remote=False)\
            as generator:
            print_time("--- %s seconds ---" % (time.time() - start_time), None)
            for prompt in prompts:
                with generator.invoke("[INST] " + prompt + " /[INST]"):
                    pass
                print_time("finished with prompt", None)
                print_time("--- %s seconds ---" % (time.time() - start_time), 
                           None)
        print_time("--- %s seconds ---" % (time.time() - start_time), None)
        model_response = tokenizer.batch_decode(generator.output)
        
        for i in range(len(prompts)):
            prompts_output.append(model_response[i].split("</s")[0]\
                                  .split("/[INST]")[-1])
            
            
    
    elif model_type == 'transformers':
        inputs = tokenizer("[INST] " + prompts[0] + " /[INST]",
                           return_tensors="pt").to(DEVICE_MAP)
        
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        for output in outputs:
            print_time("--- %s seconds ---" % (time.time() - start_time), None)
            model_response = tokenizer.decode(output,
                                              skip_special_tokens=True)
            prompts_output.append(model_response.split("</s")[0]\
                                  .split("/[INST]")[-1])

    print_time("--- %s seconds ---" % (time.time() - start_time), None)
    return(prompts_output)
    
def create_rows_prompts(cache, table, nrows):
    """
    

    Parameters
    ----------
    table : TYPE
        DESCRIPTION.
    nrows : TYPE
        DESCRIPTION.

    Returns
    -------
    prompts : TYPE
        DESCRIPTION.
    max_new_tokens : TYPE
        DESCRIPTION.

    """
    
    # TODO: Place generation of header further up in the prompt OR
    # try to say up front to generate a table in .csv file format
    # Right now this prompt fails sometimes because it does not output
    # the header. Although, if the header is still not generated every so
    # often, we can simply parse it without the header once the parsing
    # fails because we expect a header. Of course, we'd have to assume
    # the order of the output is the same as we provided in the table
    # header.

    description = table.get_description()
    header = table.get_table_header_only().to_csv(sep=table.format_type[1], 
                                                  index=False)

    table_ineligible_only = table.get_ineligible_rows_key_only(cache).to_csv(
        sep=table.format_type[1], index=False)

    delimiter = table.format_type[2]
    if nrows == 1:
        prompt = f"Generate one row for a table of {description}. "\
            + f"The {delimiter}-separated header of attributes "\
            + f"for the table is:\n{header}\n"\
            + "Do not generate fictional rows. "\
            + "Generate the rows from real known data. "\
            + f"Here is a list of {delimiter}-separated rows not to generate "\
            + f"by semantic key only:\n{table_ineligible_only}\n"\
            + "Output the row in the format of a "\
            + f"{delimiter}-separated .csv file with a column header. "\
            + "Then explain the source of the new data."
    else:
        prompt = f"Generate {nrows} rows for a table of {description}. "\
            + f"The {delimiter}-separated header of attributes "\
            + f"for the table is:\n{header}\n"\
            + "Do not generate fictional rows. "\
            + "Generate the rows from real known data. "\
            + f"Here is a list of {delimiter}-separated rows not to generate "\
            + f"by semantic key only:\n{table_ineligible_only}\n"\
            + "Output the rows in the format of a "\
            + f"{delimiter}-separated .csv file with a column header. "\
            + "Then explain the source of the new data."
    print_time(prompt, None)
    prompts = [prompt]
    # max_new_tokens = ((table.get_num_entries() + nrows + 1) * 16 
    #                   * (table.get_num_attributes() + 1) + 1000)
    max_new_tokens = 50000
    print_time(max_new_tokens, None)
    return prompts, max_new_tokens

def create_rows_from_prompts(v_cache, table, nrows, model_type, 
                             tokenizer, model, max_new_tokens, prompts):
    """
    

    Parameters
    ----------
    v_cache : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.
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
    TYPE
        DESCRIPTION.

    """
    if FAKE_MODEL:
        responses = []
        was_none = False
        if table.table is None:
            was_none = True
            table.read()
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
        resp_table = table.table.copy()
        if resp_table.shape[0] == 0:
            # our table is empty, use the original table for a source of fake 
            # rows, note that the original table was required to have at least
            # one row
            prev_table = get_table_from_cache(v_cache, table.name, 0)
            was_none_prev = False
            if prev_table.table is None:
                prev_table.read()
                was_none_prev = True
            resp_table = prev_table.table.copy()
            if was_none_prev:
                prev_table.purge()
        while resp_table.shape[0] < nrows:
            resp_table = pd.concat([resp_table, resp_table], axis=0)
        head_nrows = resp_table.head(nrows).to_csv(sep=table.format_type[1],
                                                   index=False)
        responses.append(
            f"Preamble\n\n{head_nrows}\n\nPostamble\n"
            )
        responses.append(
            f"{head_nrows}\nPreamble\n\n{head_nrows}\n\nPostamble\n"
            )
        idx = random.randrange(len(responses))
        responses = responses[idx:idx+1]
    else:
        responses = execute_prompts(model_type, tokenizer, model, 
                                    max_new_tokens, prompts)
    # first_col = list(table.table)[0]
    # print("")
    # print("first_col")
    # print_time(first_col)
        

    return parse_table_responses(table, responses, nrows)
    
def create_rows(v_cache, table_orig, nrows, model_type, 
                model, tokenizer):
    """
    

    Parameters
    ----------
    v_cache : TYPE
        DESCRIPTION.
    table_orig : TYPE
        DESCRIPTION.
    nrows : TYPE
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
    # table_orig.read()
    # if FAKE_MODEL:
    #     new_dict = {}
    #     new_dict['preamble'] = "preamble\n\n"
    #     new_dict['partial_table'] = \
    #         table_orig.table.head(nrows).to_csv(index=False)
    #     new_dict['postamble'] = "\n\npostamble\npostamble...postamble\n"
    #     return [new_dict]
    #     # return [table_orig.table.head(nrows).to_csv(index=False)]
    prompts, max_tokens = create_rows_prompts(v_cache, table_orig, nrows)
    return (prompts,
            create_rows_from_prompts(v_cache, table_orig, nrows, model_type,
                                    tokenizer, model, max_tokens, prompts))
    
def create_cols_prompts(cache, table, ncols):
    """
    

    Parameters
    ----------
    table : TYPE
        DESCRIPTION.
    ncols : TYPE
        DESCRIPTION.

    Returns
    -------
    prompts : TYPE
        DESCRIPTION.
    max_new_tokens : TYPE
        DESCRIPTION.
Generate 3 new attributes for a table of real automobile data that a purchaser 
would want to know. The semi-colon-separated header of attributes to not 
generate is
Model;Make;Year;Type;Doors;Transmission;DriveTrain;MSRP;Horsepower;Torque;MPGCity;MPGHighway;Miles;QuarterMileTime;ZeroToSixtyTime;TopSpeed
Generate real attributes. Do not generate a fictional one. Here are the 
semi-colon-separated rows of the table by semantic key only:
Model;Make;Year;Type
Camry;Toyota;2020
Corolla;Toyota;2020
Rav4;Toyota;2020
Generate values of real data for the new attributes for all existing rows of 
the table. Generate and output a new table (include the table header) with only 
the attributes Model, Make, Year, and the new attributes, in the format of a 
semi-colon-separated .csv file. Then explain the source of the new data.
    """

    description = table.get_description()

    header = table.get_ineligible_columns_header_only().to_csv(
        sep=table.format_type[1], index=False)
    
    table_key_only = table.get_table_key_only().to_csv(
        sep=table.format_type[1], index=False)

    delimiter = table.format_type[2]
    
    semantic_key = table.semantic_key

    if ncols == 1:
        prompt = f"Generate one new attributes for a table of "\
            + f"{description}. "\
            + f"The {delimiter}-separated header of attributes to not "\
            + f"generate is:\n{header}\n"\
            + "Generate a real attribute. Do not generate a fictional one. "\
            + f"Here is the {delimiter}-separated table "\
            + f"by semantic key only:\n{table_key_only}\n"\
            + "Generate values of real data for all existing rows of the "\
            + "table. "\
            + "Generate and output a new table (include the table header) "\
            + "with only the attributes "
        if len(semantic_key) > 0:
            for i in range(len(semantic_key)):
                prompt = prompt + f"{semantic_key[i]}, "
        else:
            for i in range(len(header)):
                prompt = prompt + f"{header[i]}, "
        prompt = prompt + "and the new attribute in the format of a "\
            + f"{delimiter}-separated .csv file. "\
            + "Then explain the source of the new data."
        # prompt = f"Generate one new attribute for a table of {description}. "\
        #     + f"The {delimiter}-separated header of attributes to not "\
        #     + f"generate is:\n{header}\n"\
        #     + "Generate a real attribute. Do not generate a fictional one. "\
        #     + f"Here is the {delimiter}-separated table "\
        #     + f"by semantic key only:\n{table_key_only}\n"\
        #     + "Generate values of real data for all existing rows of the "\
        #     + "table. "\
        #     + "Generate values of data for all existing rows of the table. "\
        #     + "Generate and output a new table (include the table header) "\
        #     + "with only the semantic key and the new attributes "\
        #     + f"in the format of a {delimiter}-separated .csv file. "\
        #     + "Then explain the source of the new data."
            # + "Do not output the entire table. "\
            # + "Instead, generate a new table with only the semantic key "\
            # + "and the new attribute. "\
            # + "Then explain the source of the new data."
    else:
        prompt = f"Generate {ncols} new attributes for a table of "\
            + f"{description}. "\
            + f"The {delimiter}-separated header of attributes to not "\
            + f"generate is:\n{header}\n"\
            + "Generate real attributes. Do not generate fictional ones. "\
            + f"Here are the {delimiter}-separated rows of the table "\
            + f"by semantic key only:\n{table_key_only}\n"\
            + "Generate values of real data for all existing rows of the "\
            + "table. "\
            + "Generate and output a new table (include the table header) "\
            + "with only the attributes "
        if len(semantic_key) > 0:
            for i in range(len(semantic_key)):
                prompt = prompt + f"{semantic_key[i]}, "
        else:
            for i in range(len(header)):
                prompt = prompt + f"{header[i]}, "
        prompt = prompt + "and the new attributes in the format of a "\
            + f"{delimiter}-separated .csv file. "\
            + "Then explain the source of the new data."

        # prompt = "Generate {ncols} new attributes for a table of "\
        #     + f"{description}. "\
        #     + f"The header of attributes to not generate is:\n{header}\n"\
        #     + "Generate the attributes from real known data. "\
        #     + "Here are the rows of the table by semantic key only:\n"\
        #     + f"{table_key_only}\n"\
        #     + "Generate values of real data for all existing rows of the "\
        #     + "table. "\
        #     + "Do not generate any fictional data. "\
        #     + "Generate a new table with only the semantic key "\
        #     + "and the new attributes in the format of a .csv file. "\
        #     + "Then explain the source of the new data."
            # + "Output the new attributes and their values "\
            # + "in the format of a .csv file."\
            # + "Do not output the entire table. "\
            # + "Instead, generate a new table with only the semantic key "\
            # + "and the new attributes. "\
            # + "Then explain the source of the new data."
    print_time(prompt, None)
    prompts = [prompt]
    # max_new_tokens = ((table.get_num_entries() + 1) * 16 
    #                   * (table.get_num_attributes() + 1) + 1000)
    max_new_tokens = 50000
    print_time(max_new_tokens, None)
    return prompts, max_new_tokens

def create_cols_from_prompts(v_cache, table, model_type, tokenizer,
                             model, max_new_tokens, prompts, ncols):
    """
    

    Parameters
    ----------
    v_cache : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.
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
    TYPE
        DESCRIPTION.

    """
    # table_orig.read()
    # if FAKE_MODEL:
    #     new_dict = {}
    #     new_dict['preamble'] = "preamble\n\n"
    #     new_dict['partial_table'] = \
    #         table_orig.table.head(nrows).to_csv(index=False)
    #     new_dict['postamble'] = "\n\npostamble\npostamble...postamble\n"
    #     return [new_dict]
    #     # return [table_orig.table.head(nrows).to_csv(index=False)]
    if FAKE_MODEL:
        was_none = False
        if table_orig.table is None:
            was_none = True
            table_orig.read()
        old_table = table_orig.table.copy()
        resp_table = old_table[table_orig.semantic_key].copy()
        # resp_table.drop(table_orig.semantic_key, inplace=True, axis=1)
        responses = []
        # responses.append(
        #     " Here are 2 new attributes generated for the table:"\
        #     + "1. FuelType: This attribute indicates the type of fuel used "\
        #     + "by the vehicle. The possible values are Gas, Diesel, Hybrid, "\
        #     + "and Electric. The data for this attribute is obtained from "\
        #     + "the official websites of the manufacturers or third-party "\
        #     + "automotive data providers.\n"\
        #     + "2. CityMPG: This attribute indicates the fuel efficiency "\
        #     + "of the vehicle in city driving conditions, "\
        #     + "measured in miles per gallon (MPG)."\
        #     + "The data for this attribute is obtained from the official "\
        #     + "fule economy ratings provided by the "\
        #     + "Environmental Protection Agency (EPA) of the United States."
        #     )
        if old_table.shape[1] == 0:
            # our table is empty, use the original table for a source of fake 
            # rows, note that the original table was required to have at least
            # one row
            prev_table = get_table_from_cache(v_cache, table.name, 0)
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
                
                    
            # for i, col in enumerate(old_table):
            #     print("")
            #     print("old_table")
            #     print_time(old_table)
            #     print("")
            #     print("col")
            #     print_time(col)
            #     resp_table[col + "_NEW"] = old_table[col]
            #     print("")
            #     print("resp_table")
            #     print_time(resp_table)
            #     print("")
            #     print("tot_i")
            #     print_time(tot_i)
            #     tot_i += 1
            #     if tot_i == ncols:
            #         break
        table_str = resp_table.to_csv(sep=table.format_type[1], index=False)
        responses.append(
            f"Preamble\n\n{table_str}\n\nPostamble\n"
            )
        responses.append(
            f"{table_str}\nPreamble\n\n{table_str}\n\nPostamble\n"
            
            )
        if was_none:
            table_orig.purge()
        idx = random.randrange(len(responses))
        responses = responses[idx:idx+1]
    else:
            
        responses = execute_prompts(model_type, tokenizer, model,
                                    max_new_tokens, prompts)
    # first_col = list(table.table)[0]
    # for thiscol in table.table:
    #     if thiscol in table.semantic_key:
    #         first_col = thiscol
    #         break
    return parse_table_responses(table, responses, table.table.shape[0])

# def find_valid_csv_tables(text, rows_expected, sep):
#     valid_tables = []
    
#     # start at the first line
#     text_loc = 0
#     while text_loc < len(text):
#         remain_text = text[text_loc:]
#         lines = remain_text.split("\n")
#         if len(lines) < rows_expected:
#             break
#         df = None
#         long_df = None
#         try:
#             # the table must be exactly the length that we expect
#             table_text = "\n".join(lines[:rows_expected])
#             df = pd.read_csv(io.StringIO(table_text), sep=sep)
#             print("")
#             print("df")
#             print_time(df)
#             # table exists but may be too long
#             if not df.empty:
#                 long_table_text = "\n".join(lines[:rows_expected+1])
#                 long_df = pd.read_csv(io.StringIO(long_table_text), sep=sep)
#                 # it is too long, advance text_loc to after the long table and 
#                 # continue to parse
#                 print("")
#                 print("long_df")
#                 print_time(long_df)
#                 text_loc += (len(long_table_text) + 1) # + 1 for "\n"
#                 continue
#             else:
#                 # df empty, advance to next line
#                 text_loc += (len(lines[0]) + 1) # + 1 for "\n"
#                 continue
                
#         except pd.errors.EmptyDataError:
#             if df is None:
#                 # df is None means df read failed
#                 # no valid table at this line, advaince text_loc to next line
#                 text_loc += (len(lines[0]) + 1) # + 1 for "\n"
#                 continue
#             # df read succeeded but long_df read failed, we have our table
#             begin_loc = text_loc
#             ret_table = (df, text_loc)
#             # advance text_loc to after the table
#             text_loc += (len(table_text) + 1) # + 1 for "\n"
#             ret_table = (df, begin_loc, text_loc)
#             valid_tables.append(ret_table)
#         except:
#             print_time("Exception unknown")
#             if df is None:
#                 # df is None means df read failed
#                 # no valid table at this line, advaince text_loc to next line
#                 text_loc += (len(lines[0]) + 1) # + 1 for "\n"
#                 continue
#             # df read succeeded but long_df read failed, we have our table
#             print_time(df)
#             begin_loc = text_loc
#             # advance text_loc to after the table
#             text_loc += (len(table_text) + 1) # + 1 for "\n"
#             ret_table = (df, begin_loc, text_loc)
#             valid_tables.append(ret_table)
            
#     return valid_tables 
def find_valid_csv_tables(text, nrows_expected, cols_expected, sep):
    valid_tables = []
    
    lines = text.split("\n")
    if len(lines) < nrows_expected:
        return valid_tables
    tableidx = []
    for i in range(len(lines)):
        remain = len(lines) - i
        if remain >= nrows_expected:
            found_cols = True
            for col in cols_expected:
                if col not in lines[i]:
                    found_cols = False
                    break
            if found_cols:
                tableidx.append(i)
        else:
            break
        
    for i in tableidx:
        valid = True
        for line_idx in range(i, i+nrows_expected+1):
            if len(lines[line_idx].split(sep)) < len(cols_expected):
                valid = False
                break
        if valid:
            for j in range(len(lines)):
                print_time(lines[j], None)
                lines[j] = lines[j].strip()
                print_time(lines[j], None)
            table_text = "\n".join(lines[i:i+nrows_expected+1])
            df = pd.read_csv(io.StringIO(table_text), sep=sep)
            preamble_text = "\n".join(lines[0:i])
            postamble_text = "\n".join(lines[i+nrows_expected+1:])
            valid_table = (preamble_text, df, postamble_text)
            valid_tables.append(valid_table)

    return valid_tables

# def parse_table_responses(first_col, num_rows, responses):
def parse_table_responses(table, responses, nrows_expected):
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
    print_time(nrows_expected, None)
    for response in responses:
        
        print_time(response, None)
        # time.sleep(10)
        
        preamble = ""
        postamble = ""

        # BEGIN SEARCH FOR TABLE WITHIN RESPONSE        
        # The first thing we need to do is locate our table within the
        # prompt response.
        # We have to make some assumptions here.
        #   1. Search for a table that conforms to what we asked for.
        #   2. If there's more than one, *assume it's the final table output.*
        #      This is a reasonable assumption. The chances of the table
        #      being within the prologue are far higher than the chances
        #      of the table being within the epilogue

        # Case 1: The AI has output the table with the semantic key as 
        #         instructed. So find the final table with one of the columns
        #         in the semantic key.
        valid_csv_tables = find_valid_csv_tables(response, nrows_expected,
                                                 table.semantic_key,
                                                 table.format_type[1])
        print_time(valid_csv_tables, None)

        # loop through valid tables
        col_valid_csv_table = None
        for csv_table in valid_csv_tables:
        #     if cols_expected is not None:
        #         for col in cols_expected:
        #             if col not in csv_table[0]:
        #                 continue
        #     # following is a tuple (dataframe, begin_loc, next_loc)
            col_valid_csv_table = csv_table # look for the final table
            

        # if col_valid_csv_table is None:
        #     continue
        # if col_valid_csv_table[1] > 0:
        #     preamble = response[0:col_valid_csv_table[1]]
        # if col_valid_csv_table[2] < len(response):
        #     postamble = response[col_valid_csv_table[2]:]
        # print("")
        # print("preamble")
        # print_time(col_valid_csv_table[0])
        # print("")
        # print("postamble")
        # print_time(col_valid_csv_table[2])
        # for 
        # response_idx = None
        # while True:
        #     first_idx = None
        #     first_col = None
        #     for col in table.semantic_key:
        #         idx = response.find(col)
        #         if idx is None or idx < first_idx:
        #             first_col = col
        #             first_idx = idx
        
        # # -1 because the output could have printed out the old table before the
        # # new table
        # message = response.split(first_col)
        # if len(message) == 0:
        #     print("")
        #     print_time("empty response")
        #     continue
        # if len(message) > 1:
        #     after_preamble = message[1].split('\n')
        #     num_lines_response = len(after_preamble)
        #     if num_lines_response < num_rows:
        #         num_rows_response = num_lines_response
        #     else:
        #         num_rows_response = num_rows
        #     partial_table = (first_col
        #                      + '\n'.join(after_preamble[:num_rows_response+1]))
        #     print("")
        #     print("partial_table")
        #     print_time(partial_table)
        #     if num_lines_response > num_rows:
        #         postamble = '\n'.join(after_preamble[num_rows_response+1:])
        #     else:
        #         postamble = ""
        if col_valid_csv_table is not None:
            table_response = {
                'preamble': col_valid_csv_table[0],
                'output_table': col_valid_csv_table[1].copy(),
                'postamble': col_valid_csv_table[2]
                }
            print_time(table_response, None)
            table_responses.append(table_response)
            
    return(table_responses)
    
def create_cols(v_cache, table_orig, ncols, model_type, model, tokenizer):
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
    # table_orig.read()
    # if FAKE_MODEL:
    #     new_cols = []
    #     for i, col in enumerate(table_orig.table):
    #         if i == ncols:
    #             break
    #         new_cols.append(col + "_NEW")
    #     new_table = table_orig.table.copy()
    #     for col in new_cols:
    #         new_table[col] = new_table[col[:-4]]
    #     new_table = new_table[new_cols]
    #     # return ["Crap\nCrap\nCrap\nNEW_TABLE_START\n" + new_table.to_csv(index=False)\
    #     #     + "\nNEW_TABLE_END\nCrap\nCrap\nCrap\nCrap"]
    #     new_dict = {}
    #     new_dict['preamble'] = "preamble\n\n"
    #     new_dict['partial_table'] = new_table.to_csv(index=False)
    #     new_dict['postamble'] = "\n\npostamble\npostamble...postamble\n"
    #     return [new_dict]
    prompts, max_tokens = create_cols_prompts(cache, table_orig, ncols)
    return (prompts,
            create_cols_from_prompts(v_cache, table_orig, model_type,
                                    tokenizer, model, max_tokens, prompts,
                                    ncols))
    
def fill_na_prompts(cache, table, na_loc):
    """
    

    Parameters
    ----------
    table : TYPE
        DESCRIPTION.
    ncols : TYPE
        DESCRIPTION.

    Returns
    -------
    prompts : TYPE
        DESCRIPTION.
    max_new_tokens : TYPE
        DESCRIPTION.

For a table of real automobile data that a purchaser would want to know, 
generate some real data. Do not generate fictional data. Given the following 
table, fill in the missing 'Horsepower' value corresponding to the row the 
semantic key (Model='Camry', Make='Toyota', Year='2020') and then output the 
new table with the new value of dtype int64.

Model,Make,Year,Type,Doors,Transmission,DriveTrain,MSRP,Horsepower,Torque,MPGCity,MPGHighway,Miles,QuarterMileTime,ZeroToSixtyTime,TopSpeed
Camry,Toyota,2020,Sedan,4,Automatic,Front-Wheel Drive,"$25,050",,184,28,39,39000,13.8,9.2,130
Corolla,Toyota,2020,Compact,4,CVT Automatic,Front-Wheel Drive,"$20,430",139,128,30,38,12500,12.5,7.8,110
Rav4,Toyota,2020,SUV,4,Automatic,All-Wheel Drive,"$27,285",203,184,27,34,5000,16.3,7.8,124

Then output where the data was obtained from.
    """

    description = table.get_description()
    semantic_key = table.semantic_key
    semantic_values = []
    for col in semantic_key:
        semantic_values.append(table.table.at[na_loc[1], col])
    # semantic_values_str = table.format_type[1].join(semantic_values)
    num_rows = min(table.table.shape[0], 3)
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
    col_dtype = str(table.table[attribute].dtype)
    print("")
    print("rows")
    print(rows)
    small_table = table.table.iloc[rows,:].to_csv(sep=table.format_type[1], 
                                                  index=False)
    # table_key_only = table.get_table_key_only().to_csv(
    #     sep=table.format_type[1], index=False)


    # header = table.get_ineligible_columns_header_only().to_csv(index=False)
    
    
    
    
    # header_key_only = table.get_table_key_only().head(0).to_csv(index=False)

    prompt = f"For {description}, retrieve a missing value of real data "\
        + "(not fictional) from extrnally available resources, "\
        + "corresponding to the first row and attribute "\
        + f"named {attribute}, with a dtype of {col_dtype}, "\
        + f"within the following table:\n\n{small_table}\n\n"\
        + "Retrieve the attribute value according to the "\
        + "values of the semantic key:\n\n"
    for i, key in enumerate(semantic_key):
        if na_loc[2] != key: # if the missing value is in the semantic key
            prompt = prompt + f"{key}={semantic_values[i]}\n\n"
    prompt = prompt + "Fill in the missing data, "\
        + "and output the resulting table in "\
        + f"{table.format_type[2]}-delimited .csv format. "\
        + "Then output from where the data was retrieved."

    # prompt = f"For a table of {description}, generate some real data. "\
    #     + "Do not generate fictional data.\n"
    # for na in na_list:
    #     attribute = na[1]
    #     col_dtype = str(table.table[attribute].dtype)
    #     prompt += "\nGiven the following table, fill in the missing "\
    #     + f"{attribute} value corresponding to the row with the semantic key:"\
    #     + f"\n{semantic_key} = {semantic_values_str}\n"\
    #     + "and then output the new table with the new value of dtype "\
    #     + f"{col_dtype}.\n\n{table_onerow}\n\n"
    # prompt += "Then output where all the data was obtained from."
    # if len(na_list) == 1:
    #     prompt = "Given a table of {description}, "\
    #         + "and given rows with the following values:"\
    #         + "retrieve the value for the attribute {na_list[0][1]}"
    #         + "with the following semantic key:\n{header_key_only}\n"\
    #         + "retrieve the value for the following attribute"\
    #         + f"The header of attributes to not generate is:\n{header}\n"\
    #         + "Generate the attribute from real known data. "\
    #         + "Here are the rows of the table by semantic key only:\n"\
    #         + f"{table_key_only}\n"\
    #         + "Generate values of real data for all existing rows of the "\
    #         + "table. "\
    #         + "Do not generate any fictional data. "\
    #         + "Generate and output a new table with only the semantic key "\
    #         + "and the new attribute in the format of a .csv file. "\
    #         + "Then explain the source of the new data."
    #         # + "Do not output the entire table. "\
    #         # + "Instead, generate a new table with only the semantic key "\
    #         # + "and the new attribute. "\
    #         # + "Then explain the source of the new data."
    # else:
    #     prompt = "Generate {ncols} new attributes for a table of "\
    #         + f"{description}. "\
    #         + f"The header of attributes to not generate is:\n{header}\n"\
    #         + "Generate the attributes from real known data. "\
    #         + "Here are the rows of the table by semantic key only:\n"\
    #         + f"{table_key_only}\n"\
    #         + "Generate values of real data for all existing rows of the "\
    #         + "table. "\
    #         + "Do not generate any fictional data. "\
    #         + "Generate a new table with only the semantic key "\
    #         + "and the new attributes in the format of a .csv file. "\
    #         + "Then explain the source of the new data."
    #         # + "Output the new attributes and their values "\
    #         # + "in the format of a .csv file."\
    #         # + "Do not output the entire table. "\
    #         # + "Instead, generate a new table with only the semantic key "\
    #         # + "and the new attributes. "\
    #         # + "Then explain the source of the new data."
    print_time(prompt, None)
    prompts = [prompt]
    # max_new_tokens = 4 * (table.table.shape[1] + 1)
    max_new_tokens = 50000
    return prompts, max_new_tokens

def fill_na_from_prompts(v_cache, table_orig, na_loc, model_type, tokenizer,
                         model, max_new_tokens, prompts):
    """
    

    Parameters
    ----------
    v_cache : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.
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
    TYPE
        DESCRIPTION.

    """
    # table_orig.read()
    # if FAKE_MODEL:
    #     new_dict = {}
    #     new_dict['preamble'] = "preamble\n\n"
    #     new_dict['partial_table'] = \
    #         table_orig.table.head(nrows).to_csv(index=False)
    #     new_dict['postamble'] = "\n\npostamble\npostamble...postamble\n"
    #     return [new_dict]
    #     # return [table_orig.table.head(nrows).to_csv(index=False)]
    
    if FAKE_MODEL:
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
        # semantic_values_str = table.format_type[1].join(semantic_values)
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







        # sem_val = resp_table.loc[na_loc[1], sem_key]
        # if na_loc[0] == 0:
        #     rows = [na_loc[0], 1, 2]
        # elif na_loc[0] == 1:
        #     rows = [na_loc[0], 0, 2]
        # else:
        #     rows = [na_loc[0], 0, 1]
        # if table_orig.table.shape[0] < 3:
        #     rows = rows[:resp_table.shape[0]]
        # attribute = na_loc[2]
        # col_dtype = str(resp_table[attribute].dtype)
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
            + "from the official website."\
            # " Here are 2 new attributes generated for the table:"\
            # + "1. FuelType: This attribute indicates the type of fuel used "\
            # + "by the vehicle. The possible values are Gas, Diesel, Hybrid, "\
            # + "and Electric. The data for this attribute is obtained from "\
            # + "the official websites of the manufacturers or third-party "\
            # + "automotive data providers.\n"\
            # + "2. CityMPG: This attribute indicates the fuel efficiency "\
            # + "of the vehicle in city driving conditions, "\
            # + "measured in miles per gallon (MPG)."\
            # + "The data for this attribute is obtained from the official "\
            # + "fule economy ratings provided by the "\
            # + "Environmental Protection Agency (EPA) of the United States."
            )
        # if resp_table.shape[1] == 0:
        #     # our table is empty, use the original table for a source of fake 
        #     # rows, note that the original table was required to have at least
        #     # one row
        #     prev_table = get_table_from_cache(v_cache, table.name, 0)
        #     was_none_prev = False
        #     if prev_table.table is None:
        #         prev_table.read()
        #         was_none_prev = True
        #     resp_table = prev_table.table.copy()
        #     if was_none_prev:
        #         prev_table.purge()
        # tot_i = 0
        # while tot_i < ncols:
        #     for i, col in enumerate(resp_table):
        #         resp_table[col + "_NEW"] = resp_table[col]
        #         tot_i += 1
        #         if tot_i == ncols:
        #             break
        # table_str = resp_table.to_csv(sep=table_orig.format_type[1], index=False)
        responses.append(
            f"Preamble\n\n{small_table}\n\nPostamble\n"
            )
        responses.append(
            f"{small_table}\nPreamble\n\n{small_table}\n\nPostamble\n"
            
            )
        if was_none:
            table_orig.purge()
        idx = random.randrange(len(responses))
        # idx = 0
        responses = responses[idx:idx+1]
    else:
            
        responses = execute_prompts(model_type, tokenizer, model,
                                    max_new_tokens, prompts)
    # first_col = list(table.table)[0]
    # for thiscol in table.table:
    #     if thiscol in table.semantic_key:
    #         first_col = thiscol
    #         break
    return parse_table_responses(table_orig, responses, 
                                 min(table_orig.table.shape[0], 3)) 

def fill_na(v_cache, table_orig, na_loc, model_type, model, 
            tokenizer):
    prompts, max_tokens = fill_na_prompts(cache, table_orig, na_loc)
    return (prompts, fill_na_from_prompts(v_cache, table_orig, na_loc,
                                          model_type, tokenizer, model,
                                          max_tokens, prompts))
    
# None of the following deletes should be done with inplace=True
def delete_rows(table_orig, num_entries, location):
    """
    

    Parameters
    ----------
    table_orig : TYPE
        DESCRIPTION.
    num_entries : TYPE
        DESCRIPTION.
    location : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    row_del_df : TYPE
        DESCRIPTION.

    """
    table_orig.read()
    df = table_orig.table
    # num_rows = df.shape[0]
    # if location == 'random':
    #     indicies = random.sample(list(range(num_rows)), num_entries)
    # elif location == 'top':
    #     indicies = list(range(num_entries))
    # elif location == 'bottom':
    #     indicies = list(range(num_rows - num_entries, num_rows))
    print_time(location, None)
    row_del_df = df.iloc[location,:]
    # row_del_df = row_del_df.set_index(table_orig.semantic_key, append=True)
    print_time(row_del_df, None)
    # time.sleep(10)
    idx = np.ones(len(df.index), dtype=bool)
    idx[location] = False
    return df.iloc[idx], row_del_df

def delete_cols_by_name(table_orig, cols):
    """
    

    Parameters
    ----------
    table_orig : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return table_orig.table.drop(cols, axis=1)

def delete_cols(table_orig, location):
    """
    

    Parameters
    ----------
    table_orig : TYPE
        DESCRIPTION.
    num_entries : TYPE
        DESCRIPTION.
    location : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    col_del_df : TYPE
        DESCRIPTION.

    """
    table_orig.read()
    df = table_orig.table.copy()
    
    # if location == 'random':
    #     indicies = random.sample(list(range(num_cols)), num_entries)
    # elif location == 'top':
    #     indicies = list(range(num_entries))
    # elif location == 'bottom':
    #     indicies = list(range(num_cols - num_entries, num_cols))
    print_time(location, None)
    # col_del_df = pd.DataFrame()
    col_del_list = []
    for i, col in enumerate(df):
        if i in location:
            col_del_list.append(col)
            print_time(None, f"deleting col {col} for location {i}")
            print_time(col_del_list, None)
            
            # col_del_df[col] = df[col]
    print_time(location, None)
    print_time(df, None)
    # time.sleep(10)
    col_del_df = pd.DataFrame(columns=col_del_list)
    for i, col in enumerate(df):
        if i in location:
            # col_del_list.append(col)
            print_time(None, f"deleting col {col} for location {i}")
            print_time(col_del_df, None)
            
            col_del_df[col] = df[col]
    return df.drop(df.columns[location], axis=1), col_del_df

def build_model_and_cache(model_type, model_spec, tables_folder, 
                         tables_version_delimiter):
    """
    

    Parameters
    ----------
    model_type : TYPE
        DESCRIPTION.
    model_spec : TYPE
        DESCRIPTION.
    tables_folder : TYPE
        DESCRIPTION.
    tables_version_delimiter : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.
    tokenizer : TYPE
        DESCRIPTION.
    ver_table_cache : TYPE
        DESCRIPTION.

    """
    model, tokenizer = build_model(model_type, model_spec)
    ver_table_cache = build_ver_table_cache(tables_folder, 
                                            tables_version_delimiter)
    return model, tokenizer, ver_table_cache

"""
Begin Script
"""
model, tokenizer, cache = build_model_and_cache(MODEL_TYPE, MODEL_SPEC,
                                                TABLES_FOLDER,
                                                TABLES_VERSION_DELIMITER)

used_gen = False
# ops = 0
# if FAKE_MODEL:
#     max_ops = 1
# else:
#     max_ops = 1
for idx, command_idx in enumerate(CMD_PLAN):
    # for command_idx in range(len(COMMANDS)):
    # command_idx = 4
    # command_idx = random.randrange(len(COMMANDS))
    start_time = time.time()
    new_version = get_next_ver_for_table(cache, "autona")
    print_time(new_version, None)
    if LINEAGE == "random":
        table_orig = get_table_random_from_cache(cache, "autona")
    else: # LINEAGE == "sequence"
        table_orig = get_table_from_cache(cache, "autona", 
                                          get_high_ver_for_table(cache, 
                                                                 "autona"))
    print_time(table_orig.version, None)
    print_time(table_orig.semantic_key, None)
    assert(len(table_orig.semantic_key) > 0)
    if table_orig.table is None:
        was_none = True
        table_orig.read()
    else:
        was_none = False
    prompts_output = None
    # command = COMMANDS[random.randrange(len(COMMANDS))]
    # command = COMMANDS[2]
    command = COMMANDS[command_idx]
    print_time(table_orig.semantic_key, None)
    print_time_force(command, "Starting operation...")
    command_type = command['type']
    params = command['params']
    params_list = get_params_list(params)
    random.shuffle(params_list)
    params = params_list[0]
    if command_type == "add_row":
        num_entries = params['num_entries']
        location = params['location']
        axis = 0
        nrows = table_orig.table.shape[0]
        if params['location'] == 'top':
            location = 0
        elif params['location'] == 'bottom':
            location = nrows - 1
        else:
            location = random.randrange(nrows)
        params['location'] = location
        prompts_input, prompts_output = create_rows(cache, table_orig, 
                                                    num_entries, 
                                                    MODEL_TYPE, 
                                                    model, tokenizer)
        if len(prompts_output) > 0:
            used_gen = True
        else:
            print_time(None, "add_row: Table not found!")
            time.sleep(10)
            continue
    elif command_type == "del_row":
        num_entries = params['num_entries']
        location = params['location']
        axis = 0
        nrows = table_orig.table.shape[0]
        if params['location'] == 'top':
            location = 0
        elif params['location'] == 'bottom':
            location = nrows - 1
        else:
            location = random.sample(list(range(nrows)), num_entries)
        # params['location'] = location
        new_df, changed_df = delete_rows(table_orig, num_entries, location)
        preamble = ""
        postamble = ""
        if num_entries > 1:
            deleted_row_mult_entires = True
    elif command_type == "add_col":
        num_entries = params['num_entries']
        location = params['location']
        axis = 1
        ncols = table_orig.table.shape[1]
        if params['location'] == 'left':
            location = 0
        elif params['location'] == 'right':
            location = ncols - 1
        else:
            location = random.randrange(ncols)
        # params['location'] = location
        prompts_input, prompts_output = create_cols(cache, table_orig, 
                                                    num_entries, 
                                                    MODEL_TYPE, 
                                                    model, tokenizer)
        if len(prompts_output) > 0:
            used_gen = True
        else:
            print_time(None, "add_col: Table not found!")
            time.sleep(10)
            continue
    elif command_type == "del_col":
        num_entries = params['num_entries']
        location = params['location']
        axis = 1
        indices_elig = []
        print("")
        print("table_orig.table")
        print_time(table_orig.table, None)
        for i, col in enumerate(table_orig.table):
            if col not in table_orig.semantic_key:
                indices_elig.append(i)
        ncols = table_orig.table.shape[1]
        max_indices_elig = len(indices_elig) / 2
        if num_entries > max_indices_elig:
            num_entries = min(num_entries, max_indices_elig)
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
                continue
        if params['location'] == 'left':
            location = 0
        elif params['location'] == 'right':
            location = ncols - 1
        else: # params['location'] == 'random'
            # s_len = len(table_orig.semantic_key)
            print_time(indices_elig, None)
            print_time(num_entries, None)
            location = random.sample(indices_elig, num_entries)
            print_time(location, None)
        params['location'] = location
        new_df, changed_df = delete_cols(table_orig, location)
        print_time(new_df, None)
        preamble = ""
        postamble = ""
    elif command_type == "fill_na":
        location = params['location']
        na_loc = None
        na_list = find_all_na(table_orig.table)
        if len(na_list) == 0:
            print_time_force(None, "fill_na: no N/A values to retrieve!")
            time.sleep(10)
            continue
        if params['location'] == 'random':
            random.shuffle(na_list)
        na_loc = na_list[0]
        prompts_input, prompts_output = fill_na(cache, table_orig, na_loc,
                                                MODEL_TYPE, model, tokenizer)
    
        if len(prompts_output) > 0:
            used_gen = True
        else:
            print_time_force(None, "fill_na: Table not found!")
            time.sleep(10)
            continue
        preamble = prompts_output[0]['preamble']
        table_df = prompts_output[0]['output_table']
        print_time(table_df, None)
        if table_df is None:
            print_time_force(None, "fill_na: Table not found!")
            time.sleep(10)
            continue # we did not find a table in the response, do nothing
        postamble = prompts_output[0]['postamble']
        new_df = table_orig.table.copy()
        new_df.at[na_loc[1], na_loc[2]] = table_df.at[table_df.index[0], 
                                                      na_loc[2]]
        changed_df = pd.DataFrame(columns=[na_loc[2]])
        changed_df.at[na_loc[1], na_loc[2]] = new_df.at[na_loc[1], na_loc[2]]
    if command_type == "add_row" or command_type == "add_col":
        preamble = prompts_output[0]['preamble']
        table_df = prompts_output[0]['output_table']
        output_table = table_df.to_csv(sep=table_orig.format_type[1])
        print_time(table_df, None)
        if table_df is None:
            print_time_force(None, "add_row or add_col: Table not found!")
            time.sleep(10)
            continue # we did not find a table in the response, do nothing
        postamble = prompts_output[0]['postamble']
        # if command_type == "add_col":
        #     table_df.drop(table_orig.semantic_key, axis=1, inplace=True)
        #     for col in table_df:
        #         if col in table_orig.table:
        #             table_df.drop(col, axis=1, inplace=True)
        new_df = add_table(table_orig, table_df, location, axis)
        if new_df is None:
            print_time_force(None, "Bad csv format of output")
            time.sleep(10)
            continue
        new_df = new_df.drop_duplicates()
        # changed_df = pd.read_csv(io.StringIO(tablestr), 
        #                          sep=table_orig.format_type[1]) 
        
        if (command_type == "add_row" 
              and table_df.shape[0] > table_orig.table.shape[0]):
            changed_df = pd.concat([table_df, new_df])
            changed_df = changed_df.drop_duplicates()
        else:
            changed_df = table_df.copy()
            # changed_df = changed_df.drop(table_orig.semantic_key, axis=1)
    if (command_type == "add_row" or command_type == "add_col" 
        or command_type == "fill_na"):
        begin_time = round(start_time, 0)
        end_time = round(time.time(), 0)
        duration = end_time - begin_time
        if prompts_output is not None and len(prompts_output) > 0:
            output_table = prompts_output[0]['output_table']\
                .to_csv(sep=table_orig.format_type[1])
        else:
            output_table = None
        command_dict = {'type': command_type,
                        'prompt': prompts_input[0],
                        'start time' : str(dt.datetime.fromtimestamp(
                            begin_time)),
                        'complete time': str(dt.datetime.fromtimestamp(
                            end_time)),
                        'duration (seconds)': int(duration),
                        'params': params,
                        'output_table': output_table,
                        'changed': changed_df.to_dict()}
    else:
        command_dict = {'type': command_type,
                        'params': params,
                        'changed': changed_df.to_dict()}

    new_table(cache, table_orig, TABLES_VERSION_DELIMITER, 
              preamble, new_df, postamble, command_dict,
              new_version)
    if was_none:
        table_orig.purge()
    print_time_force(command, "Finished successfully")
    # time.sleep(10)
    # ops += 1
    # if ops >= max_ops:
    #     break
"""
End script
"""