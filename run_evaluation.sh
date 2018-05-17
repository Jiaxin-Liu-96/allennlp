

# Evaluation run.
python scripts/write_srl_predictions_to_conll_format.py --path=$1 --device=$2 \
        --data=/net/efs/aristo/allennlp/srl_lm/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development/ \
        --prefix=evaluation
# Test run.
python scripts/write_srl_predictions_to_conll_format.py --path=$1 --device=$2 \
        --data=/net/efs/aristo/allennlp/srl_lm/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/test/ \
        --prefix=test

perl scripts/srl-eval.pl $1/evaluation_gold.txt $1/evaluation_predictions.txt | tee -a $1/evaluation_results.txt
perl scripts/srl-eval.pl $1/test_gold.txt $1/test_predictions.txt | tee -a $1/test_results.txt

