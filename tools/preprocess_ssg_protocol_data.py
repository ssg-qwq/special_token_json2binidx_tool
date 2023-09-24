# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys

import lm_dataformat as lmd
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import tqdm
import ftfy
import jsonlines, json
from tokenizer import build_tokenizer
import indexed_dataset
from threading import Semaphore


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = Encoder.tokenizer.tokenize(text)
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(text)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--sp_token_config",
        type=str,
        help="sp token config.",
        default="/home/li/MachineLr/json2binidx_tool/tools/sp_token_config.json",
    )
    group.add_argument(
        "--datafolder",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--jsonl-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from jsonl. Defa",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
            "RWKVTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def match_key(value, unk):
    if isinstance(unk, str):
        return value == unk
    else:
        return value in unk


def match_vs(value_to_find, unks):
    # 遍历 mapping 的 values()
    for value in unks.values():
        # 如果值是字符串，直接比较
        if isinstance(value, str):
            if value == value_to_find:
                return True
        else:
            if value_to_find in value:
                return True
    return False

def find_key_by_value(dictionary, value_to_find):
    for key, value in dictionary.items():
        if match_key(value_to_find, value):
            return key
    # 如果值不存在于字典中，可以返回None或者其他适当的默认值
    return None

def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


mapping_dict = {
    "data": "sample",  # 每一行的键
    "text": "text",  # 每一个元素的键
    "conversation": ["conversation", "query", "answer"],  # 每一个元素的键
    "system": ["system", "instruct"],  # 每一个元素的键
    "search": "search",  # 每一个元素的键
    "env": "env",  # 每一个元素的键
    "common": "common",  # 每一个元素的键
}


def main():
    args = get_args()
    with open(args.sp_token_config, 'r') as file:
        sp_token_config = json.load(file)
    encoder = Encoder(args)
    encoder.initializer()
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    dataset_name = None
    mapping = mapping_dict
    jsonl_folder = args.datafolder
    ld = os.listdir(jsonl_folder)
    config_file = os.path.basename(jsonl_folder)
    if f"{config_file}.json" in ld:
        with open(
            os.path.join(jsonl_folder, f"{config_file}.json"), "r", encoding="utf-8"
        ) as cf:
            config = json.load(cf)
            mapping.update(config)

    jsonl_list = [f for f in ld if f.endswith(".jsonl")]
    encoded_docs = []
    for jsonl_path in jsonl_list:
        with open(os.path.join(jsonl_folder, jsonl_path), "r", encoding="utf-8") as f:
            loop = tqdm.tqdm(f)
            for line in loop:
                if line.strip():
                    try:
                        loop.set_postfix(jsonl=jsonl_path)
                        data = json.loads(line)
                    except Exception as e:
                        print(f"line{line} cannot read by json:{e}")
                        continue

                    data["data"] = data.pop(mapping["data"])
                    for ele in data["data"]:
                        old_key = list(ele.keys())[0]
                        if match_vs(old_key, mapping):
                            ele[find_key_by_value(mapping, old_key)] = ele.pop(old_key)
                
                    flow = data["data"]
                    tokens=[]
                    for entity in flow:
                        role=list(entity.keys())[0]
                        token=sp_token_config[role]["prefix"]+encoder.tokenizer.tokenize(ftfy.fix_text(list(entity.values())[0]))+sp_token_config[role]["postfix"]
                        tokens+=token
                    encoded_docs.append(({"text":[tokens]},len(tokens)))
    
    # print(encoder.encode("asdasdasdasd"))
    # print(encoded_docs[-1])

    # print(len(encoded_docs))
    
    # return

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(
            args.output_prefix, key, "document"
        )
        output_idx_files[key] = "{}_{}_{}.idx".format(
            args.output_prefix, key, "document"
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
            # separate with eos token
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed:0.2f} docs/s, {mbs:0.2f} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
