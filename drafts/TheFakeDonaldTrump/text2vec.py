import os
import re

def open_text():
    try:
        file = open("data/speeches.txt","r", encoding='utf-8')
    except:
        raise ValueError("File 'speeches.txt' not found in 'data'")
    return file.read()

def split_corpus(text_as_string):
    text_as_string = text_as_string.lower()
    splitted_text = text_as_string.split("\n")
    return splitted_text

def parse(text_as_string):
    parsed_text = re.findall(r"[\w']+",text_as_string)
    return parsed_text

print(split_corpus(open_text()))