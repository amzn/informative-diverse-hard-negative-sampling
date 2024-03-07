MODEL=$1
CORPUS=$2

ls ${MODEL}
if [[ $? -ne 0 ]]; then
  echo "error, this is not a name of a model"
  exit 1
fi


echo "########## ENCODING CORPUS ##########"

mkdir -p ${MODEL}/corpus_embds
for i in $(seq -f "%02g" 0 9); do
  python -m tevatron.driver.encode \
    --output_dir ./retriever_model \
    --model_name_or_path ${MODEL} \
    --fp16 \
    --per_device_eval_batch_size 128 \
    --encode_in_path ${CORPUS}/split${i}.json \
    --encoded_save_path ${MODEL}/corpus_embds/split${i}.pt
done

echo "########## ENCODING QUERIES ##########"

python -m tevatron.driver.encode --output_dir ./retriever_model \
  --model_name_or_path ${MODEL} --fp16 --per_device_eval_batch_size 128 --q_max_len 32 --encode_is_qry \
  --encode_in_path resources/dev7k.query.json \
  --encoded_save_path ${MODEL}/query_embds.pt

echo "########## RETRIEVING ##########"

python -m tevatron.faiss_retriever \
  --query_reps ${MODEL}/query_embds.pt \
  --passage_reps ${MODEL}/corpus_embds/'*.pt' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${MODEL}/rank1k.tsv

python -m active_learning.msMarcoEval ${MODEL}/rank1k.tsv >> ${MODEL}/results.txt
