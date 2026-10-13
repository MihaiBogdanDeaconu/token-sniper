# set api key
# or set api key in `scripts/env/set_env.sh`
# . scripts/env/set_env.sh
export OPENAI_API_KEY="sk-123..."
export OPENAI_API_BASE="https://XXXXX"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR='{INDEX_DIR}/{DATASET_NAME}/graph_index_v2.3'
export BM25_INDEX_DIR='{INDEX_DIR}/{DATASET_NAME}/BM25_index'

result_path='YOUR_OUTPUT_PATH'
echo $result_path
mkdir -p $result_path

python auto_search_main.py \
    --dataset 'czlll/SWE-bench_Lite' \
    --split 'test' \
    --model 'azure/gpt-4o' \
    --localize \
    --merge \
    --output_folder $result_path/location \
    --eval_n_limit 300 \
    --num_processes 50 \
    --use_function_calling \
    --simple_desc