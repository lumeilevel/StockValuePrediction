Implements 2 models to classify stock value related texts.

The models are based on large pre-trained Foundation Models.

## BERT

1. Group Classifier: `bert.ipynb`.
2. Single Regression & Group Classification: `tfcm.ipynb`.

## Word2Vec

Group Classifier: `w2v.ipynb`.

## Result

- `prediction.py`
- `result.txt`

## Usage

To run this project:

```
python prediction.py -m/--model MODELNAME
```

where

```
MODELNAME = bert/tfcm/w2v
```

default is `tfcm`.
