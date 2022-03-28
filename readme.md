# Datasets
- [MtSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
    - Medical Transcriptions
    - Find the annotated versions by three different annotators under `data/processed/mtsamples`
- [UMLS](https://www.nlm.nih.gov/research/umls/index.html)
    - MRSCONSO.RRF file
    - Put under `data/raw`
- [ICD10 2017](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2017/)
    - icd10cm_order_2017.txt file
    - Put under `data/raw`
- [big.txt](https://norvig.com/big.txt)
    - variety of ebooks which are merged into one document
    - Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyle http://www.gutenberg.org/files/1661/1661-0.txt
    - The Project Gutenberg EBook of History of the United States by Charles A. Beard and Mary R. Beard http://www.gutenberg.org/cache/epub/16960/pg16960.txt
    - The Project Gutenberg EBook of War and Peace, by Leo Tolstoy http://www.gutenberg.org/files/2600/2600-0.txt
    - Put under `data/raw`


# Word vectors
- [Word2Vec](http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin)
    - Put under data/raw/embeddings/
- [BioBERT](https://github.com/ncbi-nlp/bluebert)
    - BERT embeddings trained on PubMed dataset

# Experiments
- Tune models: `train/tune_models.py`
    - Tunes dataset-model pairs using grid search
    - Saves results to Wandb

- Generate model performance reports: `train/run_reports.py`
    - Loads the best dataset-model pair and related config file of a trained model from wandb
    - Generates and saves results figures

# Citation
```

```