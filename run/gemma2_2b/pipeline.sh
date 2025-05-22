set -x

# Model configuration
LAYER=13
CHAT_MODEL=google/gemma-2-2b-it
BASE_MODEL=google/gemma-2-2b
TEXT_COLUMN="text"

python crosscoder_analysis_pipeline.py $@ --base-model $BASE_MODEL --chat-model $CHAT_MODEL --layer $LAYER --lmsys-col $TEXT_COLUMN --data-dir $DATASTORE --results-dir $DATASTORE/results