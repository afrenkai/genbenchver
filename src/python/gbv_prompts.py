#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:18:20 2024

@author: dfox
"""

class GenAITablePrompts:
    """
    
    """
    
    def __init__(self, cache, table, max_new_tokens):
        self.cache = cache
        self.table = table
        self.max_new_tokens = max_new_tokens
        self.prompts = []
    
    def get_fill_na_prompt(self, na_loc):
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
    
        """
    
        description = self.table.get_description()
        semantic_key = self.table.semantic_key
        semantic_values = []
        for col in semantic_key:
            semantic_values.append(self.table.table.at[na_loc[1], col])
        # semantic_values_str = self.table.format_type[1].join(semantic_values)
        num_rows = min(self.table.table.shape[0], 3)
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
        col_dtype = str(self.table.table[attribute].dtype)
        small_table = self.table.table.iloc[rows,:].to_csv(sep=self.table.format_type[1], 
                                                      index=False)
    
        prompt = f"For {description}, retrieve a missing value of real data "\
            + "(not fictional) from externally available resources, "\
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
            + f"{self.table.format_type[2]}-delimited .csv format. "\
            + "Then output from where the data was retrieved."
    
        return prompt

    def get_add_rows_prompt(self, nrows):
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
        
        description = self.table.get_description()
        
        # the following for no row examples
        header = self.table.get_table_header_only().to_csv(
            sep=self.table.format_type[1], index=False)
        
        # the following for 3 example rows
        # header = self.table.table.head(3).to_csv(
        #     sep=self.table.format_type[1], index=False)
        
        table_ineligible_only = \
            self.table.get_ineligible_rows_key_only(self.cache).to_csv(
                sep=self.table.format_type[1], index=False)
        
        delimiter = self.table.format_type[2]
        # if nrows == 1:
        #     prompt = f"Generate one row for a table of {description}. "\
        #         + f"The {delimiter}-separated header of attributes, along "\
        #         + f"with three sample rows for the table is:\n{header}\n"\
        #         + "Do not generate fictional rows. "\
        #         + "Generate the rows from real known data. "\
        #         + f"Here is a list of {delimiter}-separated rows not to "\
        #         + f"generate by semantic key only:\n{table_ineligible_only}\n"\
        #         + "Output the row in the format of a "\
        #         + f"{delimiter}-separated .csv file with a column header. "\
        #         + "Then explain the source of the new data."
        # else:
        #     prompt = f"Generate {nrows} rows for a table of {description}. "\
        #         + f"The {delimiter}-separated header of attributes, along "\
        #         + f"with three sample rows for the table is:\n{header}\n"\
        #         + "Do not generate fictional rows. "\
        #         + "Generate the rows from real known data. "\
        #         + f"Here is a list of {delimiter}-separated rows not to "\
        #         + f"generate by semantic key only:\n{table_ineligible_only}\n"\
        #         + "Output the rows in the format of a "\
        #         + f"{delimiter}-separated .csv file with a column header. "\
        #         + "Then explain the source of the new data."
        if nrows == 1:
            prompt = f"Generate one new row for a table of {description}. "\
                + f"The {delimiter}-separated header of attributes "\
                + f"for the table is:\n{header}\n"\
                + "Do not generate fictional rows. "\
                + "Generate the rows from real known data. "\
                + f"Here is a list of {delimiter}-separated rows not to "\
                + f"generate by semantic key only:\n{table_ineligible_only}\n"\
                + "Output the row in the format of a "\
                + f"{delimiter}-separated .csv file with a column header. "\
                + "Then explain the source of the new data."
        else:
            prompt = f"Generate {nrows} new rows for a table of {description}. "\
                + f"The {delimiter}-separated header of attributes "\
                + f"for the table is:\n{header}\n"\
                + "Do not generate fictional rows. "\
                + "Generate the rows from real known data. "\
                + f"Here is a list of {delimiter}-separated rows not to "\
                + f"generate by semantic key only:\n{table_ineligible_only}\n"\
                + "Output the rows in the format of a "\
                + f"{delimiter}-separated .csv file with a column header. "\
                + "Then explain the source of the new data."
        return prompt
        
    def get_add_cols_prompt(self, ncols):
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
        """
    
        description = self.table.get_description()
    
        header = self.table.get_ineligible_columns_header_only(
            self.cache).to_csv(sep=self.table.format_type[1], index=False)
        
        table_key_only = self.table.get_table_key_only().to_csv(
            sep=self.table.format_type[1], index=False)
    
        delimiter = self.table.format_type[2]
        
        semantic_key = self.table.semantic_key
    
        if ncols == 1:
            prompt = "Generate one new attribute for a table of "\
                + f"{description}. "\
                + f"The {delimiter}-separated header of attributes to not "\
                + f"generate is:\n{header}\n"\
                + "Generate a real attribute. Do not generate a fictional one. "\
                + f"Here is the {delimiter}-separated table "\
                + f"by semantic key only:\n{table_key_only}\n"\
                + "Generate values of real data for all existing rows of the "\
                + "table. "\
                + "Generate and output a new table, including the header, "\
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
                + "Generate and output a new table, include the table header, "\
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
    
        return prompt
    
    def add_prompt(self, op, **kwargs):
        
        na_loc = None
        nrows = 0
        for key, val in kwargs.items():
            if key == 'na_loc':
                na_loc = val
            if key == 'nrows':
                nrows = val
            if key == 'ncols':
                ncols = val
                
        if op == 'fill_na' and na_loc is not None:
            self.prompts.append(self.get_fill_na_prompt(na_loc))
        elif op == 'add_rows' and nrows > 0:
            self.prompts.append(self.get_add_rows_prompt(nrows))
        elif op == 'add_cols' and ncols > 0:
            self.prompts.append(self.get_add_cols_prompt(ncols))
            
        # self.prompts.append("If you have not finished displaying the previous"\
        #                     + " response, do so now.")