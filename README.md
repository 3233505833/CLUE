# CLUE

Here is the code for the LLM-based usefulness judgment method — CLUE. We used two types of LLMs.

## GPT

You can run the code of CLUE by right-clicking on  `CLUE.py`. You can switch datasets in the YAML configuration file. Additionally, we provide baseline implementations including `pointwise.py`, `pairwise.py`, and `listwise.py`, as well as machine learning method. These can also be run by right-clicking.

## LLAMA

### Step 0: Prepare Data

First, place the training and test datasets in the `zhdata` directory, and set the validation split ratio from the training set in `my.yml`.

### Step 1: Fine-Tuning

```
python trl_finetune.py -c configs/my.yml
```

### Step 2: Merge Model

```
python merge_lora.py -c configs/my.yml
```

### Step 3: Run Inference

```
python inference.py
```
