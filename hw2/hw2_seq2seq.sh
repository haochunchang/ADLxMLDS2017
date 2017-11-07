# ./hw2_seq2seq.sh $1 $2 $3
# $1: the data directory, 
# $2: test data output filename 
# $3: peer review output filename 
#Ex: ./hw2_seq2seq.sh myData  sample_output_testset.txt sample_output_peer_review.txt
if [ ! -d "./models" ]; then
  mkdir ./models
  wget --directory-prefix=models https://gitlab.com/haochunchang/ADLxMLDS2017_model_archive/raw/master/basic_models-499.data-00000-of-00001?private_token=6hvzcU4AsJbfNUGE7Ymc
  wget --directory-prefix=models https://gitlab.com/haochunchang/ADLxMLDS2017_model_archive/raw/master/basic_models-499.index?private_token=6hvzcU4AsJbfNUGE7Ymc
  wget --directory-prefix=models https://gitlab.com/haochunchang/ADLxMLDS2017_model_archive/raw/master/basic_models-499.meta?private_token=6hvzcU4AsJbfNUGE7Ymc
fi
python test_seq2seq.py $1 ./models/basic_models-499 $2 $3
