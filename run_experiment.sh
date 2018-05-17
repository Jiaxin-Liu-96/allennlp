
SERIALISATION_PATH="/net/efs/aristo/allennlp/srl_lm/emnlp/parsing/paper-analysis/"
set -e
source activate calypso
export CUDA_VISIBLE_DEVICES=$2
python allennlp/run.py train $3 -s ${SERIALISATION_PATH}$1
