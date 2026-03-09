Here is our working code for ARGOS, and an implementation of the baselines. 

In order to run, the user will have to change the path at every applicable place in the code.

Any HuggingFace pre-trained LLM can be used, and may be specificed in the arg['engine'] = line near the top of each [method]_[dataset].py script.

Pre-processing data for SAT and Backbone extraction: 

In order to obtain logical form of each problem, we run SAT-LM on the dataset, where the logic programs are written to ./SAT-LM/tmp

To process the logic into SAT, run [dataset]_to_sat.py, editing paths in the script as necessary

To obtain the SAT solutions, backbones, and FOL mappings, run cadical_solve.py and parse_gen.py in that order.

Now, argos can be run on the dataset. Make sure to edit the ./dimacs_[dataset] paths in the cot_met_[dataset] scripts under ./argos appropriately.

Also, make sure to add your huggingface personal access token in the .env file as HF_TOKEN='your_token_here'. 

Experimental results are generated in the ./analysis notebooks. 

The modified forms of each dataset can be found in ./SAT-LM/data