#  produce corresponding captions.
# $1: data dir, $2: output_file
if [ ! -d "./models" ]; then
  mkdir ./models
  wget --output-document=./models/basic_models-499.data-00000-of-00001 https://gitlab.com/haochunchang/ADLxMLDS2017_model_archive/raw/master/basic_models-499.data-00000-of-00001?private_token=6hvzcU4AsJbfNUGE7Ymc
  wget --output-document=./models/basic_models-499.index https://gitlab.com/haochunchang/ADLxMLDS2017_model_archive/raw/master/basic_models-499.index?private_token=6hvzcU4AsJbfNUGE7Ymc
  wget --output-document=./models/basic_models-499.meta https://gitlab.com/haochunchang/ADLxMLDS2017_model_archive/raw/master/basic_models-499.meta?private_token=6hvzcU4AsJbfNUGE7Ymc
fi
python special_test.py $1 ./models/basic_models-499 $2
