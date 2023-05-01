#!/usr/bin/env python3

# Get paraphrases from Turku paraphrase corpus

import sys
import json

from argparse import ArgumentParser
from logging import warning


def argparser():
    ap = ArgumentParser()
    ap.add_argument('data')
    return ap


def process(item):
    if item['label'] == '4':
        # full paraphrase
        input_ = item['txt1']
        output = item['txt2']
    elif item.get('rewrites', []):
        # rewritten to full paraphrase
        if len(item['rewrites']) != 1 or len(item['rewrites'][0]) != 2:
            warning(f'unexpected rewrites: {item["rewrites"]}')
            return
        input_ = item['rewrites'][0][0]
        output = item['rewrites'][0][1]
    else:
        # no full paraphrase
        return
    d = {
        'input': input_,
        'output': output,
    }
    print(json.dumps(d, ensure_ascii=False))


def main(argv):
    args = argparser().parse_args(argv[1:])

    with open(args.data) as f:
        data = json.load(f)

    for item in data:
        process(item)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
