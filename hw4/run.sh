#bash run.sh [testing_text.txt]
# Download pre-trained models
if [ ! -d "./models" ]; then
  mkdir ./models
  #wget --output-document=./models/latest_model.ckpt.data-00000-of-00001 https://gitlab.com/haochunchang/ADLxMLDS2017_hw4_GAN_model/raw/master/latest_model.ckpt.data-00000-of-00001?private_token=sj1NsPumWGUyMvJ6qkH2
  #wget --output-document=./models/latest_model.ckpt.index https://gitlab.com/haochunchang/ADLxMLDS2017_hw4_GAN_model/raw/master/latest_model.ckpt.index?private_token=sj1NsPumWGUyMvJ6qkH2
  #wget --output-document=./models/latest_model.ckpt.meta https://gitlab.com/haochunchang/ADLxMLDS2017_hw4_GAN_model/raw/master/latest_model.ckpt.meta?private_token=sj1NsPumWGUyMvJ6qkH2
  #wget "http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz"
  #tar -xf skip_thoughts_uni_2017_02_02.tar.gz
  #rm skip_thoughts_uni_2017_02_02.tar.gz
fi

if [ ! -d "./skip_thoughts/pretrained" ]; then
  mkdir ./skip_thoughts/pretrained
fi
#mv ./skip_thoughts_uni_2017_02_02/* ./skip_thoughts/pretrained/

python3.6 generate.py --captions $1
