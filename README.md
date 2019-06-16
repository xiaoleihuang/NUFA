# NUFA
Code Repository for the paper "Neural User Factor Adaptation for Text Classification: Learning to Generalize Across Author Demographics" at *SEM 2019.

![Image of NUFA](https://github.com/xiaoleihuang/NUFA/model.png)

## A. Data
Please download the data from [https://cmci.colorado.edu/~mpaul/files/starsem2019_demographics.data.zip](https://cmci.colorado.edu/~mpaul/files/starsem2019_demographics.data.zip)

## B. Test Environment
Ubuntu 16.04, Python 3.6+

## C. Preparations
 1. Install [conda 3.6+](https://www.anaconda.com/distribution/);
 2. Clone the repository
  * `git clone https://github.com/xiaoleihuang/NUFA.git`
  * `cd NUFA`
 3. Install packages `pip install -r requirements.txt`;
 4. Install tokenizer model, `python -c "import nltk; nltk.download('punkt')"`;
 5. Download data and unzip the data:
   * `wget https://cmci.colorado.edu/~mpaul/files/starsem2019_demographics.data.zip`
   * `unzip starsem2019_demographics.data.zip`
   * move the data: `mv ./data_hash/* data && rm -r ./data_hash/`
 6. Download pretrained embeddings:
   * You can download [Google](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and [GloVe](http://nlp.stanford.edu/data/glove.twitter.27B.zip) pretrained embeddings;
   * Our pretrained embeddings [Dropbox](https://www.dropbox.com/s/t9muudx4jrw61ge/embeddings.zip?dl=0)
   * unzip to the folder /project_folder/w2v/

## D. Instructions for analysis sections (2.2 and 2.3):
 * 2.2 Are User Factors Encoded in Text?
   * 2.2.1 User Factor Prediction
     * `cd document_predictability`
     * `python demographic_clf.py`
     * back to the project root folder: `cd ..`
   * 2.2.2 Topic Analysis
     * `cd topic`
     * Build topic models: `python build_model.py`
     * Calculate log ratios across demographic groups: `python viz_ratio.py`
     * Images will be saved to the ./ratios/ folder.
     * `cd ..`
 * 2.3 Are Document Categories Expressed Differently by Different User Groups?
  * `cd word_overlap`
  * `python cal_mi.py`
  * You can change the size of top features


## E. How to run
### Experiment Setups
  1. Split the data into train/valid/test sets:
    * `cd data`
    * `python data_split.py`
  2. Build tokenizer:
    * `cd tokenizer`
    * `python build_tok.py`
    * `cd ..`
  3. Convert raw data into indices of words:
    * `python data2indices.py`
  4. Build initialized weights for embeddings:
    * `cd weight`
    * `python build_wt.py`
  5. Build vectorizers for non-neural models:
    * `cd /path_to_data_folder/`
    * `python fea_builder.py`

### Baselines
  1. N-gram:
    * `cd no_ngram`
    * `python LR_3gram.py`
  2. CNN
    * `cd no_cnn`
    * `python Kim_CNN_keras.py`
  3. Bi-LSTM
    * `cd bilstm`
    * `python BiLSTM.py`
  4. FEDA
    * `cd daume`
    * `python build_vects_clfs.py`
  5. DANN
    * `cd dann`
    * single domain: `python DANN_keras_1.py`
    * multi domains: `python DANN_keras_sample_multi_domain_cnn.py`

### NUFA
  1. NUFA
    * `cd nufa`
    * single domain:
      * `python DANN_keras_sample_single_domain_lstm3.py`
      * You can manually remove the adversarial training.
      * no-shared bi-lstm: `python DANN_keras_sample_single_domain_lstm3_noshared.py`
  2. NUFA-all
    * `python DANN_keras_sample_multi_domain_lstm3.py`
    * You can manually remove the adversarial training.
  3. NUFA-weighted
    * `python DANN_keras_sample_multi_domain_lstm3_weighted.py`

## G. Contact and Citation
Contact by Email: [xiaolei.huang@colorado.edu](mailto:xiaolei.huang@colorado.edu)

```
@inproceedings{huang-paul-2019-neural,
    title = "Neural User Factor Adaptation for Text Classification: Learning to Generalize Across Author Demographics",
    author = "Huang, Xiaolei  and
      Paul, Michael J.",
    booktitle = "Proceedings of the Eighth Joint Conference on Lexical and Computational Semantics (*{SEM} 2019)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-1015",
    pages = "136--146",
    abstract = "Language use varies across different demographic factors, such as gender, age, and geographic location. However, most existing document classification methods ignore demographic variability. In this study, we examine empirically how text data can vary across four demographic factors: gender, age, country, and region. We propose a multitask neural model to account for demographic variations via adversarial training. In experiments on four English-language social media datasets, we find that classification performance improves when adapting for user factors.",
}
```
