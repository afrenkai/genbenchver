#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:19:20 2024

@author: dfox
"""

import os
import json
import pandas as pd
from gbv_utils import print_time

class VerTable:
    """
    
    """
    
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
    
    def print_debug(self, var, text):
        if self.debug:
            print_time(self, var, text)
    
    def __init__(self, table, folder, name, description, preamble, postamble, 
                 semantic_key, version_delimiter, format_type, version, 
                 lineage, command_dict, debug):
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
        self.debug = debug
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
                    # if FAKE_MODEL:
                    #     if 'semantic_key' not in json_dict:
                    #         self.semantic_key = ["Model, Make, Year"]
                    # else:
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
    
    def get_ineligible_columns_header_only(self, cache):
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
            table = VerTable.get_table_from_cache(cache, self.name, i)
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
            table = VerTable.get_table_from_cache(cache, self.name, i)
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
