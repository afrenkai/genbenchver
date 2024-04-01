#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:19:20 2024

@author: dfox
"""

import os
import json
import pandas as pd
import random
import sys
from gbv_utils import print_time

class VerTableCache:
    
    def __init__(self, folder, **kwargs):
        self.this = {}
        self.folder = folder
        self.version_delimiter = "_"
        self.supported_formats = [".csv"]
        self.debug = False
        self.new_format_type = (".csv", ";", "semi-colon")
        
        for key, value in kwargs.items():
            if key == 'version_delimiter':
                self.version_delimiter = value
            if key == 'supported_formats':
                self.supported_formats = value
            if key == 'new_format_type':
                self.new_format_type = value
            if key == 'debug':
                self.debug = value
                
        for filename in os.listdir(self.folder):
            # for now, only csv supported
            filetype = "." + filename.split(".")[-1]
            self.print_debug(filetype, None)
            self.print_debug(self.supported_formats, None)
            if filetype in self.supported_formats:
                fullname = filename.split(".")[0]
                s = fullname.split(self.version_delimiter)
                if len(s) == 1:
                    name = s[0]
                    version = None
                elif len(s) >= 2:
                    if s[-1].isdecimal():
                        name = self.version_delimiter.join(s[0:-1])
                        version = int(s[-1])
                    else:
                        name = self.version_delimiter.join(s)
                        version = None
                table = VerTable(None, name, self.folder, version, None,
                                 version_delimiter=self.version_delimiter,
                                 debug=self.debug)
                if table is not None and version is not None:
                    self.add(table)
                if version is None:
                    if table.convert(self.new_format_type):
                        self.add(table)
        
    def add(self, table):
        if table.name not in self.this:
            self.this[table.name] = {}
        name_dict = self.this[table.name]
        name_dict[table.version] = {'table': table}
        
    def get(self, name, version):
        if name in self.this:
            if version in self.this[name]:
                return self.this[name][version]['table']
        return None

    def get_high_ver_for_table(self, name):
        return max(list(self.this[name].keys())) 
    
    def get_missing_ver_for_table(self, name):
        ver_list = list(self.this[name].keys())
        for i in range(max(ver_list)):
            if i not in self.this[name]:
                return i
        return max(ver_list) + 1
    
    def get_next_ver_for_table(self, name):
        return self.get_high_ver_for_table(name) + 1
    
    def get_table_random_from_cache(self, name):
        if name in self.this:
            r = random.randrange(len(self.this[name]))
            for i, version in enumerate(self.this[name]):
                if i == r:
                    table = self.this[name][version]['table']
                    if table is not None:
                        return table
        return self.get_next_ver_for_table(name)
                                
    
    def read_table_from_cache(self, name, version):
        table = self.get(name, version)
        if table is not None:
            table.read()
        return table
    
    def print_debug(self, var, text):
        if self.debug:
            print_time(var, text)
    

class VerTable:
    """
    
    """
    
    def print_debug(self, var, text):
        if self.debug:
            print_time(var, text)
    
    def __init__(self, table, name, folder, version, info, **kwargs): 
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
        prologue : TYPE
            DESCRIPTION.
        epilogue : TYPE
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
        self.name = name
        self.folder = folder
        self.version = version
        
        if info is not None:
            self.description = info['description']
            self.lineage = info['lineage']
            self.semantic_key = info['semantic_key']
            self.format_type = (info['file_ext'], info['field_delim'],
                                info['file_ext_name'])
            

        # kwarg defaults
        self.version_delimiter = "_"
        self.create_command = None
        self.debug = False
        
        # kwarg processing
        for key, value in kwargs.items():
            if key == 'version_delimiter':
                self.version_delimiter = value
            if key == 'format_type':
                self.format_type = value
            if key == 'create_command':
                self.create_command = value
            if key == 'debug':
                self.debug = value
        
        if self.version is not None:
            self.version_str = self.version_delimiter + str(self.version)
        
        if info is None or 'lineage' not in info or info['lineage'] is None:
            self.lineage = []
        else:
            self.lineage = info['lineage']
            
        if self.version is not None:
            self.filespec = os.path.join(self.folder, (self.name 
                                                       + self.version_str))
        else:
            self.filespec = os.path.join(self.folder, self.name)
        
        json_dict = None
        if info is None:
            self.description = ""
            if os.path.exists(self.filespec + ".json"):
                with open(self.filespec + ".json", "r") as fp:
                    json_dict = json.load(fp)
                    self.print_debug(json_dict, None)
                    self.description = json_dict['description']
                    # if 'lineage' in json_dict:
                    self.lineage = json_dict['lineage']
                    # if FAKE_MODEL:
                    #     if 'semantic_key' not in json_dict:
                    #         self.semantic_key = ["Model, Make, Year"]
                    # else:
                    self.semantic_key = json_dict['semantic_key']
                    # if 'prologue' in json_dict:
                    #     self.prologue = json_dict['prologue']
                    # if 'epilogue' in json_dict:
                    #     self.epilogue = json_dict['epilogue']
                    self.format_type = (json_dict['file_ext'], 
                                        json_dict['field_delim'],
                                        json_dict['file_ext_name'])
                    if 'command' in json_dict:
                        self.command_dict = json_dict['command']
            
        if json_dict is None:
            json_dict = {}
            if info is not None:
                json_dict['description'] = self.description
                json_dict['lineage'] = self.lineage
                json_dict['semantic_key'] = self.semantic_key
                json_dict['file_ext'] = self.format_type[0]
                json_dict['field_delim'] = self.format_type[1]
                json_dict['file_ext_name'] = self.format_type[2]
            if self.create_command is not None:
                json_dict['command'] = self.create_command
            with open(self.filespec + ".json", 'w') as fp:
                json.dump(json_dict, fp, indent=4)
            
        if type(self.table) == str:
            self.write()
        elif self.table is not None: # type is dataframe
            self.write()

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
    
    def convert(self, new_format_type):
        self.print_debug(new_format_type, "convert")
        new_version = 0
        new_version_str = self.version_delimiter + str(new_version)
        new_filespec = os.path.join(self.folder, (self.name + new_version_str 
                                                  + new_format_type[0]))
        if os.path.exists(new_filespec):
            return False
        old_filespec = os.path.join(self.folder, (self.name 
                                                  + self.format_type[0]))
        new_json_filespec = os.path.join(self.folder, (self.name + new_version_str 
                                                       + ".json"))
        orig_delim = self.format_type[1]
        new_delim = new_format_type[1]

        df = pd.read_csv(old_filespec, sep=orig_delim)
        df.to_csv(new_filespec, sep=new_delim, index=False)

        json_dict = {}
        json_dict['description'] = self.description
        json_dict['lineage'] = self.lineage
        json_dict['semantic_key'] = self.semantic_key
        json_dict['file_ext'] = new_format_type[0]
        json_dict['field_delim'] = new_format_type[1]
        json_dict['file_ext_name'] = new_format_type[2]
        
        with open(new_json_filespec, 'w') as fp:
            json.dump(json_dict, fp, indent=4)
            
        self.version = new_version
        self.version_str = new_version_str
        
        return True

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
            # self.num_entries = self.table.shape[0]
            # self.num_table_attributes = self.table.shape[1]
            self.table.reset_index(drop=True, inplace=True)
            # self.table.to_csv(filename, sep=self.format_type[1], index=False)
            return True
        return False
    
    def purge(self):
        self.table = None
        
    # def get_num_entries(self):
    #     was_none = self.read()
    #     num_entries = self.table.shape[0]
    #     if was_none:
    #         self.purge()
    #     return num_entries
    
    # def get_num_attributes(self):
    #     was_none = self.read()
    #     num_entries = self.table.shape[1]
    #     if was_none:
    #         self.purge()
    #     return num_entries
    
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
            table = cache.get(self.name, i)
            if table.create_command is not None:
                if table.create_command['type'] == 'del_col':
                    for col in table.create_command['changed']:
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
            table = cache.get(self.name, i)
            if (table.create_command is not None 
                and table.create_command['type'] == 'del_row'):
                extend_df = pd.DataFrame(table.create_command['changed'])
                if len(self.semantic_key) == 0:
                    extend_dfsk = extend_df.copy()
                else:
                    extend_dfsk = extend_df[self.semantic_key].copy()
                extend_dfsk = extend_dfsk.drop_duplicates()
                self.print_debug(extend_dfsk, None)
                self.print_debug(dfsk, None)
                dfsk = pd.concat([dfsk, extend_dfsk])
                dfsk = dfsk.drop_duplicates()
        # dfsk = df[self.semantic_key].copy()
        # dfsk = dfsk.drop_duplicates()
        return dfsk
    
