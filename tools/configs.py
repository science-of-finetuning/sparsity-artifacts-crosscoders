
MODEL_CONFIGS = {
    "Qwen/Qwen2.5-1.5B": {
        "ignore_first_n_tokens_per_sample": 21,
        "text_column": "text_qwen2_5",
        "attn_implementation": None,
        "token_level_replacement": None,
    },
    "google/gemma-2-2b": {
        "ignore_first_n_tokens_per_sample": 0,
        "text_column": "text",
        "attn_implementation": "eager",
        "token_level_replacement": None,
    },
}
MODEL_CONFIGS["google/gemma-2-2b-it"] = MODEL_CONFIGS["google/gemma-2-2b"]
MODEL_CONFIGS["Qwen/Qwen2.5-1.5B-Instruct"] = MODEL_CONFIGS["Qwen/Qwen2.5-1.5B"]
