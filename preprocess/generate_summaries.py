import argparse
import json
import re
from summarizer import Summarizer

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_utt_ids(in_text, in_sum):
    utt_id = {}
    id = 0
    for utt in in_text:
        utt_id[utt] = id
        id += 1
    output = []
    for utt in tokenizer.tokenize(in_sum):
        # for some reason, summarizer removes first bracket in [PERSONxx]
        # and adds random spaces or brackets or other stuff
        utt = re.sub("\[ PERSON", "[PERSON", utt)
        utt = re.sub("\[ ", "", utt)
        utt = re.sub("\( ", "", utt)
        if utt.startswith("PERSON"):
            utt = '[' + utt
        utt = re.sub("But\)", "(But)", utt)
        if utt != "[":
            output.append(utt_id[utt])

    return output


def create_candidates(text_file, num_utt):
    in_data = load_jsonl(text_file)
    with open("cand_index.jsonl", 'w') as out_file:
        for item in in_data:
            body = " ".join(item["text"])
            model = Summarizer()
            result = model(body, num_sentences=num_utt)
            # now convert it to set ids
            candidate = get_utt_ids(item["text"], result)

            json.dump({"sent_id": candidate}, out_file)
            out_file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process truncated documents to obtain candidate summaries'
    )
    parser.add_argument('--text', type=str, required=True,
                        help='Path to file with original data')
    parser.add_argument('--num_sent', type=int, required=True,
                        help='Number of sentences for extraction')

    args = parser.parse_args()
    create_candidates(args.text, args.num_sent)