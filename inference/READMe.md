# base attention extend cases

- add batching + casual mask
- add KV cache resue with decode
    - add page attention
- add MQA
- profiling
- add TP sharding
- add flash attention implementation
- add rope


01_mha_baseline.py（batch + causal mask）

02_kv_cache_decode.py（prefill/decode 分离）

03_gqa_mqa.py

04_paged_kv.py（block_table + gather）

05_blockwise_attention.py（online softmax 思路）

bench.py（统一 benchmark + profiler）