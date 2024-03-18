#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dfox
"""
import os
import pandas as pd
import random
import time
import datetime as dt
import io
import json

# Note only comma-delimited csv files are supported at this time

TABLES_FOLDER = "tables"
TABLES_VERSION_DELIMITER = "_"
TABLES_DESC_SPEC = "_desc.txt"
MODEL_SPEC = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
MODEL_TYPE = 'nnsight'

FAKE_MODEL = False

if MODEL_TYPE == 'transformers':
    from transformers import AutoModelForCausalLM, AutoTokenizer
elif MODEL_TYPE == 'nnsight':
    from nnsight import LanguageModel
    import torch.nn as nn

def update_ver_table_cache(vtindex, name, table, version):
    if name not in vtindex:
        vtindex[name] = {}
    name_dict = vtindex[name]
    name_dict[version] = {'table': table}
    
def build_ver_table_cache(folder, version_delimiter):
    cache = {}
    # get base files first
    for filename in os.listdir(folder):
        # for now, only csv supported
        if filename.endswith(".csv"):
            fullname = filename.split(".")[0]
            s = fullname.split(version_delimiter)
            if len(s) == 1:
                name = s[0]
                version = 0
                # json_fn = name + ".json"
                # if os.path.exits(json_fn):
                    # with open(filename) as fp:
                    #     info = json.load(fp)
                    # if info is not None:
                table = VerTable(None, folder, name, None, (".csv", ","), 
                                 version, [])
                if table is not None:
                    update_ver_table_cache(cache, name, table, version)
    
    # # now get versioned files
    # for filename in os.listdir(folder):
    #     # for now, only csv supported
    #     if filename.endswith(".csv"):
    #         fullname = filename.split(".")[0]
    #         s = fullname.split(version_delimiter)
    #         if len(s) == 2:
            elif len(s) >= 2 and s[-1].isdecimal():
                name = version_delimiter.join(s[0:-1])
                version = int(s[-1])
                # json_fn = name + ".json"
                # if os.path.exits(json_fn):
                    # table = VerTable(None, folder, name, info['description'],
                    #                  (".csv", ","), version, 
                    #                  info['lineage'])
                    # update_ver_table_cache(vtindex, name, table, version)
                table = VerTable(None, folder, name, None, (".csv", ","), 
                                 version, [])
                if table is not None:
                    update_ver_table_cache(cache, name, table, version)
    return cache

def set_node_ver_table_cache(vtindex, table):
    update_ver_table_cache(vtindex, table.name, table, table.version)
                
def unset_node_ver_table_cache(vtindex, table):
    update_ver_table_cache(vtindex, table.name, None, table.version)

def get_next_ver_for_table(cache, name):
    new_version = max(list(cache[name].keys())) + 1
    # Note: base_table must be in the cache, it's a critical error if it's not
    # so it's wholly appropriate to crash here if that's the case
    # new_version = base_table.version.copy()
    # cur_ver_dict = vtindex[base_table.name]
    # print("")
    # print("cur_ver_dict")
    # print(cur_ver_dict)
    # next_ver = str(int(base_table.version[0]) + 1)
    # print("")
    # print("next_ver")
    # print(next_ver)
    # if next_ver in cur_ver_dict:
    #     print(f"{next_ver} in cur_ver_dict")
    #     for v in new_version:
    #         cur_ver_dict = cur_ver_dict[v]
    #     if v not in cur_ver_dict:
    #         new_version.append("1")
    # else:
    #     new_version[-1] = next_ver
    return new_version

# def get_next_ver_for_table(cache, base_table):
#     cur_ver_dict = cache[base_table.name]
#     get_next_ver(cur_ver_dict, base_table.version)

# def get_next_ver(cname_dict, base_version):
#     cur_ver_dict = cname_dict
#     for v in base_version:
#         cur_ver_dict = cur_ver_dict[v]
#     next_ver = str(int(base_version[-1]) + 1)
#     if next_ver in cur_ver_dict:
#         # branching
#         base_version.append("1")
#         return get_next_ver(cname_dict, base_version)

# def get_filename_from_name_version(folder, name, version, format_type):
#     fn = f"{name}"
#     for v in version:
#         fn = fn + f"{TABLES_VERSION_DELIMITER}{v}"
#     print(fn)
#     print(f"{format_type[0]}")
#     fn = fn + f"{format_type[0]}"
#     print(fn)
#     return os.path.join(folder, fn)

def get_table_from_cache(cache, name, version):
    if name in cache:
        if version in cache[name] and 'table' in cache[name][version]:
            table = cache[name][version]['table']
            if table is not None:
                return table
    return None

def read_table_from_cache(cache, name, version):
    table = get_table_from_cache(cache, name, version)
    if table is not None:
        table.read()
    return table
    #     filename = name + TABLES_VERSION_DELIMITER + str(version) + format_type[1]
    #     filespec = os.path.join(folder, filename)
    #     if os.path.exits(filespec):
    #         if name in 
    # print("")
    # print("version")
    # print(version)
    # if name not in cache:
    #     print("")
    #     print("")
    #     print(f"{name} not in cache")
    #     return None
    # name_dict = cache[name]
    # print(name_dict)
    # if name_dict['table'] is None:
    #     print("reading table")
    #     # table = VerTable(pd.read_csv(
    #     #     get_filename_from_name_version(folder, name, version, 
    #     #                                    format_type)), 
    #     #     TABLES_FOLDER, name, f"{TABLES_DESC_SPEC}", format_type, version)
    #     with open(os.path.join(folder, f"{name}{TABLES_DESC_SPEC}")) as fp:
    #         desc = fp.read()[:-1]
    #     print("")
    #     print("version")
    #     print(version)
    #     table = VerTable(None, folder, name, desc, format_type, version)
    #     print("")
    #     print("version")
    #     print(version)
    #     print(table)
    #     cur_ver_dict['table'] = table
    # print("")
    # print("")
    # print("vtindex from get_table_from_cache")
    # print(vtindex)
    # print("")
    # print("cur_ver_dict['table'].version")
    # print(cur_ver_dict['table'].version)
    # return cur_ver_dict['table']

def add_table_to_cache(vtindex, table):
    if table.name not in vtindex:
        vtindex[table.name] = {}
    cur_ver_dict = vtindex[table.name]
    if table.version not in cur_ver_dict:
        cur_ver_dict[table.version] = {}
    
    if 'table' in cur_ver_dict[table.version]:
        print("table already set in cache! Overwriting...")
    cur_ver_dict[table.version]['table'] = table

class VerTable:
    # NEXT_TODO: read from json (if description=None), write to json
    def __init__(self, table, folder, name, description, format_type, version,
                 lineage):
        self.table = table
        self.folder = folder
        self.name = name
        self.description = description
        self.format_type = format_type
        self.version = version
        if self.version == 0:
            self.version_str = ""
        else:
            self.version_str = "_" + str(self.version)
        if lineage is None:
            lineage = []
        self.lineage = lineage
        print(self.name)
        print(self.version_str)
        self.filespec = os.path.join(self.folder, self.name + self.version_str)
        # if self.version == ['0']:
        #     self.version_str = ""
        # else:
        #     self.version_str = TABLES_VERSION_DELIMITER\
        #         + TABLES_VERSION_DELIMITER.join(self.version)
        # self.filespec = os.path.join(self.folder, f"{self.version_str}"\
        #                              + "{self.format_type[0]}")
        # if self.version == ['0']:
        #     self.base_version = None
        # elif self.version == ['1']:
        #     self.base_version = ['0']
        # elif self.version[-1] == '1':
        #     self.base_version = self.version[:-2].copy()
        # else:
        #     self.base_version = self.version.copy()
        #     self.base_version[-1] = str(int(self.version[-1]) - 1)
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
                    print(json_dict)
                    self.description = json_dict['description']
                    self.lineage = json_dict['lineage']
            
        if json_dict is None:
            json_dict = {}
            json_dict['description'] = self.description
            json_dict['lineage'] = self.lineage
            print("")
            print("json_dict")
            print(json_dict)
            with open(self.filespec + ".json", 'w') as fp:
                json.dump(json_dict, fp)
            
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
        if self.table is not None:
            return self.table.to_csv(index=False)
        print("returning 'None'")
        return "None"
    
    def write(self):
        # for now, csv only format type supported
        filename = self.filespec + self.format_type[0]
        if type(self.table) == str:
            with open(filename, "w") as fp:
                fp.write(self.table)
            self.table = pd.read_csv(filename, sep=self.format_type[1])
        else:
            print("")
            print("")
            print(self.format_type[1])
            self.table.to_csv(filename, sep=self.format_type[1],
                              index=False)
            
    def read(self):
        filename = self.filespec + self.format_type[0]
        if self.table == None:
            self.table = pd.read_csv(filename, sep=self.format_type[1])
            print("just read table")
            print(self.table)
            self.num_entries = self.table.shape[0]
            self.num_table_attributes = self.table.shape[1]
        
    def get_num_entries(self):
        return self.table.shape[0]
    
    def get_num_attributes(self):
        return self.table.shape[1]
    
    def update(self, table):
        self.table = table
        self.write()
    
def new_table(cache, orig_table, new_df, new_version):
    # if type(data) == str:
    #     df = pd.read_csv(io.StringIO(data), sep=orig_table.format_type[1]) # lineterminator='\n'
    # else: # dataframe
    #     df = data 
    # print("")
    # print("df")
    # print(df)
    # print("")
    # print("orig_table.table")
    # print(orig_table.table)
    # new_df = pd.concat([orig_table.table, df], axis=axis)
    # print("")
    # print("new_version")
    # print(new_version)
    # print("")
    # print("new_df")
    # print(new_df)
    # print("")
    # print("")
    # print(orig_table.format_type[1])
    lineage = orig_table.lineage
    if lineage is None:
        lineage = []
    lineage.append(orig_table.version)
    new_table = VerTable(new_df, orig_table.folder, orig_table.name,
                         orig_table.description, orig_table.format_type,
                         new_version, lineage)
    add_table_to_cache(cache, new_table)
    return new_table

def add_table(orig_table, data, axis):
    if type(data) == str:
        df = pd.read_csv(io.StringIO(data), sep=orig_table.format_type[1]) # lineterminator='\n'
    else: # dataframe
        df = data 
    print("")
    print("df")
    print(df)
    print("")
    print("orig_table.table")
    print(orig_table.table)
    new_df = pd.concat([orig_table.table, df], axis=axis)
    print("")
    print("new_df")
    print(new_df)
    return new_df
        
def build_model(model_type, model_id):
    if model_type == 'nnsight': 
        model = LanguageModel(model_id, device_map='auto')
        tokenizer = model.tokenizer
        
    elif model_type == 'transformers':
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
    return model, tokenizer

def execute_prompts(model_type, tokenizer, model, max_new_tokens, prompts):
    random.seed(42)
    start_time = time.time()
    start = dt.datetime.now()
    print("")
    print("")
    print(f"--- starting at {start}")
    for promptin in prompts:
        print("")
        print("")
        print("INPUT")
        print("")
        print(promptin)
        print("")
        print("END INPUT")
        print("")
    prompts_output = []
    
    if model_type == 'nnsight':
        with model.generate(max_new_tokens=max_new_tokens, remote=False)\
            as generator:
            print("--- %s seconds ---" % (time.time() - start_time))
            for prompt in prompts:
                with generator.invoke("[INST] " + prompt + " /[INST]"):
                    pass
                print("finished with prompt")
                print("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))
        # print(generator.output)
        for i in range(len(prompts)):
            # model_response = tokenizer.batch_decode(generator.output)[i]
            model_response = tokenizer.decode(generator.output[i])
            print("")
            print("")
            print("MODEL RESPONSE")
            print("")
            print(model_response)
            print("")
            print("END MODEL RESPONSE")
            print("")
            print("")
            # prompts_output.append(model_response.split("</s")[0]
            #                       .split("/[INST]")[0].split("[INST]")[1])
            prompts_output.append(model_response.split("</s")[0]
                                  .split("/[INST]")[-1])
            
            
    
    elif model_type == 'transformers':
        inputs = tokenizer("[INST] " + prompts[0] + " /[INST]",
                           return_tensors="pt")
        
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        print("--- %s seconds ---" % (time.time() - start_time))
        model_response = tokenizer.decode(outputs[0],
                                          skip_special_tokens=True)
        # print(model_response)
        prompts_output = [model_response.split("/[INST] </s>")[-1]
            .split("<s> [INST]")[1]]

    # print(prompt_output)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
    print("")
    for promptout in prompts_output:
        print("")
        print("")
        print("OUTPUT")
        print("")
        print(promptout)
        print("")
        print("END OUTPUT")
        print("")
    return(prompts_output)
        
# def create_from_prompts(v_cache, table, model_type, tokenizer, model, 
#                              max_new_tokens, prompts):
#     response = execute_prompts(model_type, tokenizer, model, max_new_tokens, 
#                                prompts)
#     print(response)
    
#     table_new = VerTable(response, table.folder, table.name, table.description, 
#                           (".csv", ","), 
#                           get_next_ver_from_cache(v_cache, table))
#     return(table_new)
    
def create_table_prompt_from_description(name, table_description, 
                                         ncols, cols_description,
                                         nrows, rows_description,
                                         index_include):
    """
    Create a new table of real automobile data, which is 26 rows by 16 columns. 
    The first row is a column header. The first 15 columns have attributes that 
    an automobile purchaser would like to know. The final column is titled 
    "Deleted" which contains a value of 0 for each of the 25 rows. Each of the 
    25 rows of instances consist of randomly selected top-selling models. 
    Format the table using | as the column delimiter. Decimal values will use a 
    period and not a comma. Output the entire table of 26 rows and only the 
    entire table of 26 rows. Do not output any explanation.

    Notes: Do not assume only the table is output.
    Tokenize first on |.
    Then strip out any beginning or ending token up to a final line feed.

    Parameters
    ----------
    name : TYPE
        DESCRIPTION.
    table_description : TYPE
        DESCRIPTION.
    ncols : TYPE
        DESCRIPTION.
    cols_description : TYPE
        DESCRIPTION.
    nrows : TYPE
        DESCRIPTION.
    rows_description : TYPE
        DESCRIPTION.
    index_include : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    prompt = f"Create a table of {table_description}. It has {ncols} columns of "\
        + f"attributes{cols_description} and {nrows} rows of instances"\
        + f"{rows_description}. Format the table exactly as a .csv file, "\
        + "delimited by commas. Required decimal values will use a period "\
        + f"and not a comma. Output the entire table of {nrows} rows and only "\
        + f"the entire table of {nrows} rows."
    if index_include:
        prompt = prompt + " The first column should contain a monotonically "
        + "increasing index."
    # Change the multiplier below as necessary. There is no magic to this.
    # It is merely our best-guess as to how many new tokens will need
    max_tokens = ncols * (nrows+1) * 30 
    return prompt, max_tokens

def create_table_from_prompt(model_type, tokenizer, model, max_new_tokens, 
                             folder, name, table_description, prompt):
    response = execute_prompts(model_type, tokenizer, model, max_new_tokens, 
                               [prompt])
    
    print(response)
    
    return VerTable(response, folder, name, table_description, 
                    (".csv", ","), 0, [])
        
def create_auto_table_from_description(model_type, tokenizer, model, folder, 
                                       nrows, ncols):
    
    table_description = "real automobile data that an automobile purchaser"\
        + " would like to know"
    c_prompt, max_tokens = create_table_prompt_from_description(
        "auto",
        table_description,
        ncols, "",
        nrows, " of top-selling models",
        False)
    print(c_prompt)
    return create_table_from_prompt(model_type, tokenizer, model, max_tokens, 
                                    folder, "auto", table_description, 
                                    c_prompt)
    
# response = "Placeholder"
def create_auto_table(cache, model_type, model, tokenizer):
    # this code creates an automobile table from scratch
    auto_orig = create_auto_table_from_description(model_type, tokenizer, model,
                                                   TABLES_FOLDER, 25, 15)
    # auto_orig = VerTable(table, table.folder, table.name, (".csv", ","), None, [0])
    set_node_ver_table_cache(cache, auto_orig)
    
    print(cache)
    # auto_orig.write()
    
    #     "filespec)
    # auto_top_selling_create_info = {
    #     'table_description': f"of real automobile data",
    #     'ncols': 15, # includes index or key
    #     'cols_description': "that an automobile purchaser would like to know",
    #     'nrows': 25, # includes header if specified
    #     'rows_description': "top-selling models"
    #     'index_include': "should not",
    #     }
    return auto_orig
    
def create_rows_prompts(table, nrows):

    # tablestr = str(table)
    if nrows == 1:
        prompt = "Generate one new non-duplicate row for a table of "\
            + f"{table.description}. The current table is:\n\n {str(table)}\n"\
            + "\nOutput the newly generated row with header and no other "\
            + "text. Do not output the full table, just the new rows, except "\
            + "output RESPONSE_START on one line prior to the output of the "\
            + "new rows and one line of RESPONSE_END after the output of the "\
            + "new rows"
    else:
        prompt = f"Generate {nrows} new non-duplicate rows for a table of "\
            + f"{table.description}. The current table is:\n\n{str(table)}\n"\
            + f"\nOutput the {nrows} newly generated rows with header and no "\
            + "other text. Do not output the full table, just the new rows, except "\
            + "output RESPONSE_START on one line prior to the output of the "\
            + "new rows and one line of RESPONSE_END after the output of the "\
            + "new rows"
    prompts = [prompt]
    # input_table_entries = len(tablestr.split("\n"))
    # input_table_attributes = len(tablestr.split("\n")[0].split(","))
    # input_tokens = input_table_entries * input_table_attributes * 3 + 150
    # output_tokens = (nrows+1) * input_table_attributes * 3
    # max_new_tokens = input_tokens + output_tokens
    max_new_tokens = table.get_num_attributes() * 4 * (nrows + 1)
    print("")
    print("")
    print(f"max_new_tokens {max_new_tokens}")
    print("")
    print("")
    # * len(table.table.columns)
    return prompts, max_new_tokens

def create_rows_from_prompts(v_cache, table, model_type, tokenizer, model, 
                             max_new_tokens, prompts):
    response = execute_prompts(model_type, tokenizer, model, max_new_tokens, 
                               prompts)
    return response
    # response = execute_prompt(model_type, tokenizer, model, max_new_tokens, 
    #                           prompt)
    # print(response)
    
    # table_new = VerTable(response, table.folder, table.name, table.description, 
    #                      (".csv", ","), 
    #                      get_next_ver_from_cache(v_cache, table))
    # return(table_new)
    
def create_rows(v_cache, table_orig, nrows, model_type, model, tokenizer):
    if FAKE_MODEL:
        # return "NEW_TABLE_START", "NEW_TABLE_END",\
        #     ["Crap\nCrap\nCrap\nRESPONSE_START\n"\ 
        #      + table_orig.table.head(nrows).to_csv(index=False)\
        #     + "\nRESPONSE_END\nCrap\nCrap\nCrap\nCrap"]
        return [table_orig.table.head(nrows).to_csv(index=False)]
    prompts, max_tokens = create_rows_prompts(table_orig, nrows)
    return create_rows_from_prompts(v_cache, table_orig, model_type,
                                    tokenizer, model, max_tokens, prompts)
    
def add_rows(table_orig, nrows, location):
    
    pass

"""
The attached table is of real automobile data that a purchaser would want to 
know. Generate a new attribute/column that is relevant and generate values for 
that column for all existing rows.  Output the entire table, including all 
rows, but no columns except the new one.
"""

def create_cols_prompts(table, ncols):

    # tablestr = str(table)
    # input_tokens_col = len(tablestr.split("\n"))
    if ncols == 1:
        prompt = \
            f"""The following table is of {table.description}:\n{str(table)}. 
            Generate one new attribute/column that is relevant and generate 
            values for that column for all existing rows.  Output only the 
            entire table, including all rows, but no columns except the new 
            one. Do not output any other text, except output RESPONSE_START 
            on one line prior to the output of the new table and one line of 
            RESPONSE_END after the output of the new table.
            """
    else:
        prompt = \
            f"""The following table is of {table.description}:\n{str(table)}. 
            Generate {ncols} new attributes/columns that are relevant and 
            generate values for those columns for all existing rows.  Output 
            only the entire table, including all rows, but no columns except 
            for the {ncols} new columns. Do not output any other text, except
            output RESPONSE_START on one line prior to the output of the
            new table and one line of RESPONSE_END after the output of the
            new table.
            """
    prompts = [prompt]
    # input_table_entries = len(tablestr.split("\n"))
    # input_table_attributes = len(tablestr.split("\n")[0].split(","))
    # input_tokens = input_table_entries * input_table_attributes * 3 + 200
    # output_tokens = (ncols+1) * input_table_entries * 3
    # max_new_tokens = input_tokens + output_tokens
    max_new_tokens = table.get_num_entries() * 4 * (ncols + 1)
    print("")
    print("")
    print(f"max_new_tokens {max_new_tokens}")
    print("")
    print("")
    return prompts, max_new_tokens

def create_cols_from_prompts(v_cache, table, model_type, tokenizer, model, 
                             max_new_tokens, prompts):
    response = execute_prompts(model_type, tokenizer, model, max_new_tokens, 
                               prompts)
    table_start = response[0].split("RESPONSE_START")[1]
    # table_start = table_start.split('\n')
    # print("")
    # print("table_start")
    # print(table_start)
    # tablestr = "\n".join(table_start[0:num_entries+1])
    return table_start.split("RESPONSE_END")[0]
    # response = execute_prompt(model_type, tokenizer, model, max_new_tokens, 
    #                           prompt)
    # print(response)
    
    # table_new = VerTable(response, table.folder, table.name, table.description, 
    #                      (".csv", ","), 
    #                      get_next_ver_from_cache(v_cache, table))
    # return(table_new)
    
def create_cols(v_cache, table_orig, ncols, model_type, model, tokenizer):
    if FAKE_MODEL:
        new_cols = []
        for i, col in enumerate(table_orig.table):
            if i == ncols:
                break
            new_cols.append(col + "_NEW")
        new_table = table_orig.table.copy()
        for col in new_cols:
            new_table[col] = new_table[col[:-4]]
        new_table = new_table[new_cols]
        # return ["Crap\nCrap\nCrap\nNEW_TABLE_START\n" + new_table.to_csv(index=False)\
        #     + "\nNEW_TABLE_END\nCrap\nCrap\nCrap\nCrap"]
        return [new_table.to_csv(index=False)]
    prompts, max_tokens = create_cols_prompts(table_orig, ncols)
    return create_cols_from_prompts(v_cache, table_orig, model_type,
                                    tokenizer, model, max_tokens, prompts)
    
def add_cols(table_orig, cols, location):
    pass

# None of the following deletes should be done with inplace=True
def delete_rows(table_orig, indicies):
    return table_orig.table.drop(index=indicies)

def delete_random_rows(table_orig, num_entries):
    indicies = []
    indicies.append(random.randrange(table_orig.table.shape[1]))
    print("")
    print("indices")
    print(indicies)
    return delete_rows(table_orig, indicies)

def delete_cols_by_name(table_orig, cols):
    return table_orig.table.drop(cols, axis=1)

def delete_cols_by_index(table_orig, indicies):
    return table_orig.table.drop(table_orig.table.columns[indicies], axis=1)

def delete_random_cols(table_orig, num_entries):
    indicies = []
    indicies.append(random.randrange(table_orig.shape[0]))
    print("")
    print("indices")
    print(indicies)
    return delete_cols_by_index(table_orig, indicies)

"""
The following table is of {table.description}:\n{str(table)}. Fill in any NaN 
values with reasonable data. Please output only the fixed row(s) and no other 
text.
"""
def build_model_and_cache(model_type, model_spec, tables_folder, 
                         tables_version_delimiter):
    print("")
    print("build_model")
    model, tokenizer = build_model(model_type, model_spec)
    
    ver_table_cache = build_ver_table_cache(tables_folder, 
                                            tables_version_delimiter)
    print("")
    print("ver_table_cache")
    print(ver_table_cache)
    return model, tokenizer, ver_table_cache

print("")
print("build_model_and_cache")
model, tokenizer, cache = build_model_and_cache(MODEL_TYPE, MODEL_SPEC,
                                                TABLES_FOLDER,
                                                TABLES_VERSION_DELIMITER)

print("")
print("")
print(model)
# print(ver_table_cache)
# model, tokenizer = build_model(MODEL_TYPE, MODEL_SPEC)

# ver_table_cache = build_ver_table_cache(TABLES_FOLDER, 
#                                         TABLES_VERSION_DELIMITER)
# print(ver_table_cache)
        
# auto_orig = create_auto_table(MODEL_TYPE, model, tokenizer)
auto_orig = read_table_from_cache(cache, "auto", 1) 
print("")
print("auto_orig.version")
print(auto_orig.version)
print("")
print("")
print(cache)
new_version = get_next_ver_for_table(cache, "auto")
print("")
print("new_version")
print(new_version)
# auto_new = add_row(ver_table_cache, auto_orig, MODEL_TYPE, model, tokenizer)

# print(create_rows(ver_table_cache, auto_orig, 1, MODEL_TYPE, model, tokenizer))
# print(create_cols(ver_table_cache, auto_orig, 5, MODEL_TYPE, model, tokenizer))

COMMAND_TYPE = "add_rows"
if COMMAND_TYPE == "add_rows":
    axis = 0
    num_entries = 1
    prompts_output = create_rows(cache, auto_orig, num_entries, MODEL_TYPE, 
                                 model, tokenizer)
elif COMMAND_TYPE == "del_rows":
    axis = 0
    num_entries = 5
    new_df = delete_random_rows(auto_orig, num_entries)
elif COMMAND_TYPE == "add_cols":
    axis = 1
    num_entries = 1
    prompts_output = create_cols(cache, auto_orig, num_entries, MODEL_TYPE, 
                                 model, tokenizer)
elif COMMAND_TYPE == "del_cols":
    axis = 0
    num_entries = 5
    new_df = delete_random_cols(auto_orig, num_entries)

if COMMAND_TYPE == "add_rows" or COMMAND_TYPE == "add_cols":
    print("")
    print("prompts_output")
    print(prompts_output)
    # auto_orig.read()
    # first_col = list(auto_orig.table.columns)[0]
    # first_col = auto_orig.table.columns[0]
    # print("")
    # print("first_col")
    # print(first_col)
    # # support only one prompt for now
    # table_start = first_col + prompts_output[0].split(first_col)[1]
    # table_start = table_start.split('\n')
    # print("")
    # print("table_start")
    # print(table_start)
    # tablestr = "\n".join(table_start[0:num_entries+1])
    tablestr = prompts_output[0]
    print("")
    print("tablestr")
    print(tablestr)
    new_df = add_table(auto_orig, tablestr, axis)
auto_new = new_table(cache, auto_orig, new_df, new_version)
print("")
print("auto_new")
print(auto_new)