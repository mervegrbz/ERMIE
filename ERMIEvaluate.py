# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import spacy_udpipe
import pandas as pd
import copy

from MWEPreProcessor import MWEPreProcessor
from WordEmbedding import set_fastText_word_embeddings
from Operations import load_pickle, get_logger, dump_pickle
from MWEIdentifier import MWEIdentifier

class ERMIEvaluate:
    def __init__(self, root_path):
        self.lang = 'TR'
        self.tag = 'gappy-crossy'
        self.embedding_type = 'headend'
        self.root_path = root_path
        self.input_path = self.root_path
        self.output_path = os.path.join(self.root_path, 'output', self.lang)
        gensim_name = "gensim_" + self.lang.lower()
        self.gensim_we_path = os.path.join(self.root_path, 'TR_model/Embeddings', gensim_name)

        self.mwe_write_path = os.path.join(self.root_path, "output")

        if not os.path.exists(self.mwe_write_path):
            os.makedirs(self.mwe_write_path)

        self.mwe_train_path = os.path.join(self.root_path, 'TR_model', 'train.pkl')

        self.logger = get_logger(os.path.join(self.root_path, 'TR_model'))

        self.params = { 'TR': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 20},
                  }

        self.mwe = load_pickle(self.mwe_train_path)
        
        self.mwe_identifier = MWEIdentifier(self.lang, self.embedding_type, self.mwe, self.logger, self.mwe_write_path)
        self.mwe_identifier.set_params(self.params[self.lang])
        self.mwe_identifier.set_train()
        self.mwe_identifier.build_model()
        
    def evaluate(self, sentence):
        
        nlp = spacy_udpipe.load_from_path(lang="tr",
                                      path="./turkish-imst-ud-2.4-190531.udpipe",
                                      meta={"description": "Custom 'tr' model"})
        text = sentence

        doc = nlp(text)
        udpiped_sentence = [(token.i + 1, token.text, token.lemma_, token.pos_, "_", "_", str(token.head), token.dep_.lower(), "_", "_", "_") for token in doc]
        self.mwe.test_sentences = [udpiped_sentence]
        new_corpus = pd.DataFrame(udpiped_sentence, columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL',
                                                   'DEPS', 'MISC', 'PARSEME:MWE'])
        new_corpus['BIO'] = copy.deepcopy(new_corpus['PARSEME:MWE'])
        new_corpus[new_corpus['BIO'].isnull()] = 'space'
        new_corpus['BIO'] = copy.deepcopy(new_corpus['BIO'].apply(lambda x: x.strip()))
        space_row = {'ID':'space', 'FORM':'space', "LEMMA":'space', "UPOS":'space', "XPOS":'space', "FEATS":'space', "HEAD":'space', "DEPREL":'space', "DEPS":'space', "MISC":'space', "PARSEME:MWE":'space', "BIO":'space'}
        test_corpus = new_corpus.append(space_row, ignore_index=True)
        self.mwe._test_corpus = test_corpus
                
        self.mwe_identifier.mwe = self.mwe
        self.mwe_identifier.set_test()
        reload_path = os.path.join(self.root_path, 'TR_model', 'teacher-weights-last.hdf5')
        lines = self.mwe_identifier.predict_test_custom_model(reload_path)
        
        return lines
