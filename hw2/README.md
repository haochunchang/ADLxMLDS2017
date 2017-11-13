# HW2-Video Caption Generation
* Task: Generate caption for given videos

## Package Requirement:
* pandas (0.21.0)
* numpy (1.12.1)
* tensorflow (r1.3)

## Assume data directory tree
* ./data/
    * bias\_init\_vector.npy
    * ixtoword.npy
    * wordtoix.npy
    * peer\_review\_id.txt
    * testing\_data
        * feat
    * testing\_id.txt
    * testing\_label.json
    * training\_data
        * feat
    * training\_label.json

## Training From Scratch
```
python model_seq2seq.py [data directory]
# or python3 model_seq2seq.py [data directory]

```
## Model Architecture
* For basic seq2seq model, see [VCG\_model.py](./VCG_model.py)
_Note: this script may produce warning about "state is tuple == True", this will be deprecated in the future_

* For attention-based model, see [VCG\_atten\_model.py](./attention/VCG_atten_model.py)
* For GRU-cell seq2seq model, see [VCG\_model\_gru.py](./VCG_model_gru.py)

## Testing
```
# Basic seq2seq, bleu score = 0.2648 / 0.5753
./hw2_seq2seq.sh [data directory] [outfilepath] [peer review output]

```

