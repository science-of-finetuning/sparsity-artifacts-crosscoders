set -x

# Model configuration
LAYER=16
CHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
BASE_MODEL=meta-llama/Meta-Llama-3.1-8B
TEXT_COLUMN="text_llama3"

python crosscoder_analysis_pipeline.py $@ --base-model $BASE_MODEL --chat-model $CHAT_MODEL --layer $LAYER --lmsys-col $TEXT_COLUMN --data-dir $DATASTORE --results-dir $DATASTORE/results