# HW2-Video Caption Generation
* Task: Generate caption for given videos

## Package Requirement:
* pandas (0.21.0)
* numpy (1.12.1)
* keras (2.0.7)
* tensorflow (r1.3)
* scikit-learn (0.19.0)

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
* see [VCG\_model.py](./VCG_model.py)

## Testing (Basic seq2seq, bleu score = 0.257)
```
./hw2_seq2seq.sh [data directory] [outfilepath] [peer review output]

```

