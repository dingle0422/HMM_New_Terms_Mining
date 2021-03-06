# HMM_New_Terms_Mining (Chinese Only)
## Mining new terms in specific domains for the downstream NLP or other algorithm work

1. corpus_prep.py -- Set up the initial data preparation process and define the logic of the BMES marker.
2. Matrix_generator.py -- Generate the transfit matrix, emit matrix, and head matrix (h-mat is unnecessary yet) based on the corpus marked within BMES symbol.
3. HMM_Model.py -- Based on the matrix, mining the new terms in the testing corpus using viterbi. Users may use this to do term-mining sentence by sentence to fit some downstreaming NLP work
4. word_filter.py -- Handle some word-filterring work like short/long terms' overlap instances and low-document-frequency instances
5. main.py -- This is the method to mining new terms with a whole corpus document, so the model will dig out all the underlay proper new terms within a big set
6. odd_handle.py -- Used to check some odd new-terms, to inspect how the words located in all related sentence, what the attributes of components of the new-terms are. Thus, users can add some artificial new-terms by hand, so the model's corpus knowledge will be more sufficient.(This can be regarded as a remedial method)


## config.txt description
### data_name:
1. old: The domain terms we already have, to sharpen our jieba tokenizer (path: ./data)
2. new: The new corpus file you want to dig the new-terms, originally its in pd.dataframe format (path: ./data)
3. corp_col: The column name of corpus in the pd.dataframe file
4. corp_file: The file name of the corpus cleaned by code 1.

### strategy:
1. keep_stopwords: boolean, to direct the model whether keep the stopwords. Once you change the parameter, please rerun the whole workflow 1. to 2.
2. threshold: int, to set up the threshold number of overlap-words filter

## The whole workflow will be upgraded continuously. For now, its good to go.
