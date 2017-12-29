#bash run.sh [testing_text.txt]
# Download pre-trained models
if [ ! -d "./models" ]; then
  mkdir ./models
  wget --output-document=./models/latest_model.ckpt.data-00000-of-00001 https://gitlab.com/haochunchang/ADLxMLDS2017_hw4_GAN_model/raw/master/latest_model.ckpt.data-00000-of-00001?private_token=sj1NsPumWGUyMvJ6qkH2
  wget --output-document=./models/latest_model.ckpt.index https://gitlab.com/haochunchang/ADLxMLDS2017_hw4_GAN_model/raw/master/latest_model.ckpt.index?private_token=sj1NsPumWGUyMvJ6qkH2
  wget --output-document=./models/latest_model.ckpt.meta https://gitlab.com/haochunchang/ADLxMLDS2017_hw4_GAN_model/raw/master/latest_model.ckpt.meta?private_token=sj1NsPumWGUyMvJ6qkH2
fi

python3.6 generate.py --captions $1
