# jsonl to binidx tool（using special token）

This repository is greatly simplified from https://github.com/EleutherAI/gpt-neox, to ONLY convert .jsonl into .bin and .idx , can serve for dataset preparation of RWKV model (see https://github.com/BlinkDL/RWKV-LM), 

Motivation: datasets (-> special_token_json2binidx_tool -> RWKV-LM(-infctx) pretraining) -> base LLM -> https://github.com/neromous/RWKV-Ouroboros online training

## Using rwkv-4-world models tokenizer rwkv_vocab_v20230424.txt.
```
python tools/preprocess_ssg_protocol_data.py --datafolder ./your_folder --sp_token_config ./tools/sp_token_config.json --output-prefix ./data/sample --vocab ./rwkv_vocab_v20230424.txt --dataset-impl mmap --tokenizer-type RWKVTokenizer --append-eod
```

The data folder strucutre:
```
-your_folder
  -a.jsonl
  -b.jsonl
  -your_folder.json
```

The jsonl format sample (one line for each document):
```json
{"sample": [{"aaa":"以下是一段对话"},{"conversation":"Question: 你是谁？"},{"conversation":"Answer:阿巴阿巴，我也不知道我是谁"},{"system":"Answer后是AI的回答"},{"conversation":"Question: 你是谁？"},{"conversation":"Answer:我是AI，这是我的回答。"}]}
```

mapping rule config "your_folder.json" format sample:
```json
{
    "data": "sample", # represents mapping "data" field in ssg-protocol into "sample"
    "text": "aaa", # represents mapping "text" special tokens in sp_token_config.json into "aaa"
    "conversation": "conversation",
    "system": "system"
}
```

for non-sp-token datasets:
```json
{"sample": [{"text":"content of dataset"}]}
```
