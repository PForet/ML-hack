from __future__ import division
import re
import numpy as np
from collections import Counter


def open_text():
    try:
        file = open("data/speeches.txt","r")
    except:
        raise ValueError("File 'speeches.txt' not found in 'data'")
    return file.read().decode('utf-8').lower()

def parse(text_as_string):
    parsed_text = re.findall(r"[\w']+|\.|,",text_as_string)
    return parsed_text

class lexicon:
    
    def __init__(self, parsed_text):
        self.parsed_text = parsed_text
        self.count = Counter(self.parsed_text)
        self.N_ = len(self.parsed_text)
        self.corpus = set(self.parsed_text)
        self.forge_dictionnary()
        
    def forge_dictionnary(self):
        corpus_as_list = list(self.corpus)
        self.encoder = {word:number for word,number in zip(corpus_as_list, range(len(self.corpus)))}
        self.decoder = {number:word for word,number in zip(corpus_as_list, range(len(self.corpus)))}
        self.encoder["_OTHER_"] = len(self.corpus)
        self.decoder[len(self.corpus)] = "_OTHER_"
        
    def keep_most_common(self, n):
        reduced = self.count.most_common(n)
        kept = sum([j for i,j in reduced])
        self.corpus = set([i for i,j in reduced])
        print("{} words kept over {}.".format(kept, self.N_))
        print("Residuals : {} %".format((1-kept/self.N_)*100))
        self.forge_dictionnary()
        
    def keep_above(self, n):
        reduced = [(i, self.count[i]) for i in self.count if self.count[i] >= n]
        kept = sum([j for i,j in reduced])
        self.corpus = set([i for i,j in reduced])
        print("{} words kept over {}.".format(kept, self.N_))
        print("Residuals : {} %".format((1-kept/self.N_)*100))
        self.forge_dictionnary()

    def get_sequence(self, start, length):
        
        def word_to_vector(word):
            if word not in self.encoder.keys(): word = "_OTHER_"
            indx = self.encoder[word]
            tmp = np.zeros(len(self.corpus)+1)
            tmp[indx] = 1
            return tmp
        
        return [word_to_vector(i) for i in self.parsed_text[start:start+length]]
        
    def decode(self, sequence):
        indx_sequence = np.argmax(sequence, axis=1)
        return map(lambda x: self.decoder[x], indx_sequence)
    
mylex = lexicon(parse(open_text()))