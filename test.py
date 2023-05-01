#!/usr/bin/env python3

import sys
import json

import torch

from argparse import ArgumentParser

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from train import PREFIX


def argparser():
    ap = ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('prompts')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if torch.cuda.is_available():
        model.to('cuda')
    print('model on', model.device)
    
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=model.device
    )

    with open(args.prompts) as f:
        for l in f:
            data = json.loads(l)
            input_ = PREFIX + data['input'] + '\n\n'
            output = pipe(
                input_,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_beams=20,
                num_return_sequences=20,
                no_repeat_ngram_size=2,
                repetition_penalty=0.9,
                max_new_tokens=100
            )
            print(input_, end='')
            uniq = []
            for o in output:
                g = o['generated_text'][len(input_):]
                if g != data['input'] and g not in uniq:
                    uniq.append(g)
            for g in uniq:
                print(g)
            print('-' * 10)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
