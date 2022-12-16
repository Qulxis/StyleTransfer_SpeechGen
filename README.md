# Repository for Language Generation and Summarization Project:
Description: This project aims to build a constrained generative model along the axes of politeness using exisitng politeness strategy models. 

## Installation and setup
1. Install requirements.txt
```
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
```
2. Create a wandb account for recording model results and performance issues.

## Use
The main APIs to gather and preprocess tweets are contianed in "Experimentation/api/". A walk through of this process is in "Experimentation/Scrapping and Cleaning.ipynb". It uses snscrape to gather data and then processes the raw data for suitable use. Baselines for politness are available through runing baselines_politeness which is the demo file from https://github.com/CornellNLP/ConvoKit/blob/master/examples/politeness-strategies/politeness_demo.ipynb to check that the API is functional for this instance.
Full_model_processing.ipynb completes the whole processing step post scrapping and cleaning. Once it is run, GPT2_model_new.ipynb is used to generate fined tuned gpt-2 models. The scoring for this output was done manually by human annotators located at https://docs.google.com/spreadsheets/d/1p1CCXQranUoIe5MO4VuEUfSCtboTBzyQIKcRcezktQ0/edit#gid=1528582561.

