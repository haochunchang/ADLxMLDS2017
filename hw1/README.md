# HW1-Sequence Labeling
#### Task: Label each frames obtained from speech signal
#### DataSet: TIMIT

## Package Requirement:
pandas (0.21.0)
numpy (1.12.1)
keras (2.0.7)
tensorflow (r1.3)
scikit-learn (0.19.0)

## Assume data directory tree

data/
----fbank/
--------test.ark
--------train.ark
----label/
--------train.lab
----mfcc/
--------test.ark
--------train.ark
----phones/
--------48\_39.map
----48phone\_char.map

## Testing pre-trained model
```
#RNN
./hw1_rnn.sh data/ [output_file]

#CNN+RNN
./hw1_cnn.sh data/ [output_file]

#Best
./hw1_best.sh data/ [output_file]

```

## Training data Preprocessing
* Please run the preprocessing script before training from scratch.
```python
python sequence_data.py
```

## Others
* label\_map.pkl: Label mapping using sklearn LabelBinarizer

