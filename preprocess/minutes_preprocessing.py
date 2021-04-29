import os
import argparse
import json
import re

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def readfile(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


def preprocess(in_text):
    '''
    Add spaces between some punctuation to improve tokenization
    :param in_text: input text
    :return: input text with extra spaces
    '''
    output = re.sub('([\.\?,;-])([a-zA-Z\[\(<])', '\\1 \\2', in_text)
    # this is to fix issues when candidate summaries are generated
    output = re.sub('\(\?\?\)', '', output)
    output = re.sub('<mmhm>', '', output)
    output = re.sub('<ehm>', '', output)
    output = re.sub('  ', ' ', output)
    return output


def process_minutes(input_files):
    text = {}
    summary = {}

    for subdir, dirs, files in os.walk(input_files):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filename.startswith("transcript"):
                text[subdir] = readfile(filepath)
            elif filename.startswith("minutes"):
                summary[subdir] = readfile(filepath)

    assert len(text) == len(summary)

    with open("all_original_data.jsonl", 'w') as out_file:
        for dir, txt in text.items():
            txt = preprocess(txt)
            sum_split = summary[dir].split('- ')
            if '•	' in summary[dir]:
                sum_split = summary[dir].split('•	')
            json_str = {
                "text": tokenizer.tokenize(txt),
                "summary": sum_split
            }
            json.dump(json_str, out_file)
            out_file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process truncated documents to obtain candidate summaries'
    )
    parser.add_argument('--path', type=str, required=True,
                        help='Path  to folders containing input files')

    args = parser.parse_args()
    process_minutes(args.path)
