# Self-Attentive Sequential Recommendation model

A plain implementation of [SASRec](https://ieeexplore.ieee.org/abstract/document/8594844) model, as part of a thesis on LLM-based recommendation.

You can access the model from [MongrelIntruder/sasrec-pytorch](https://huggingface.co/MongrelIntruder/sasrec-pytorch) once fully implemented.

The dataset used for the training is [movielens](https://grouplens.org/datasets/movielens/) 1m entries (2003)

## How to run

Simply install the dependencies and run `main.py`

1. Create an environment and activate it

```bash
python -m venv venv
```

```bash
source venv/bin/activate
# Windows: venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the program

```bash
python main.py
```
