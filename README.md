# genbenchver
Generative Table Versioning Benchmarks

This repository is the code and created benchmarks corresponding to the following publication:

Daniel C. Fox, Aamod Khatiwada, Roee Shraga, "A Generative AI Benchmark Creation Framework for Detecting Common Data Table Versions."
Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM), October 2024.

Files:
    
    src/python/gbv_main.py: 
        Main routine and supporting source code for executing benchmark creation
    src/python/gbv_ver_table.py: 
        VerTable class:
            Instantiation for each specific version of a table in the entire series of benchmarks.
            In general, some instantiations have loaded the entire table into a pandas dataframe, but most do not.
        VerTableCache class:
            Singleton that maintains a cache of all versions of all tables in a VerTable object.
            Although all versions of all tables are in this cache, most versions of tables will not have the contents
            of their table loaded. If the calling routine needs access to the table, it can call a read() method,
            followed up with a purge() when it is done with it.
    src/python/gbv_prompt.py:
        GenAITablePrompts class:
            Routines to generate prompts for each request.
    src/python/gbv_utils.py:
        Minor utility functions.
    src/python/gbv_convert.py:
        Routine to convert a table in one file format to the one needed for the benchmark creation.
        Runs stand-alone.
    src/python/gbv_create.py:
        NOT USED. Only in repository as an example of how you would create a table with an LLM.
        Code is very preliminary and should only be used as a template for the code that ends up implementing
        table create.

    tables/<name>_<version>.csv:
        Semi-colon delimited file containing table of a specific <name> and <version>

    tables/<name>_<version>.json:
        Metadata for each table. Metadata explanations TBD.

The benchmarks currently consist of 20 versions of each of five small seed tables.

Seed tables are required to be in semi-colon separated format without leading or trailing whitespace in each cell.
gbv_convert.py can be used to do the conversion, but you need to identify filenames, original separator, etc.
Note that if you have a different format than one that is included in this file, then you may need to augment the code.

The --debug mode is a little difficult to follow (sorry for that, too).

# Command-line Interface:

## Run the main() routine in gbv_main.py (-h or --help for help):
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
                            is 3.



# Future Work: Use Hosted LLM API
  
    In gbv_main.py:
      Modify main():
        Define a new framework (-f command-line prompt) for each LLM API (i.e. pplxapi_mixtral8x7b)
      Modify build_model():
        Initialize the LLM API here. model will probably be a returned handle.
        Not sure if tokenizer will be meaningful
      Modify GenAITableExec.execute_prompts():
        This method takes in a list of prompts to execute, along with the corresponding prompts
          *More than one prompt has not been tested and almost certainly doesn't work*
        Given framework, put your API call in here. Returns text of response.
    Modify gbv_prompts.py:
      Modify GenAITablePrompts.__init__():
        You propbaly want to either pass in the framework or the name of the LLM or the handle 
        of the model in order to identify which model/framework you are using. This is because 
        the prompt engineering may be slightly different from model/framework to 
        model/framework. 
        
        For example, the Mixtral 8x7b API might require a different prompt 
        than the nnsight framework, just because I used a temperature of 0.1 for nnsight, 
        and I have no idea what the hosted code will use (though you may be able to find out). 
        
        The point is, you just might need different prompts if you switch to the API and/or 
        the LLM model.
      Modify GenAITablePrompts.get_fill_na_prompt(), get_add_rows_prompt(), get_add_cols_prompt()
        Again, you may need to use a different prompt template for the API framework.
    I think that's it.
