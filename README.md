# DiffusEmp
This repo contains the official implementation for the submission	paper 2587: DiffuEmp: A Conditional Diffusion Language Model with Multi-Grained Control for Empathetic Response Generation


## Highlights
The main contribution of this paper is threefold:
(1) We introduce explicit multi-grained control signals to solve the monotonous empathy problem, and convert the empathetic response generation into a controllable setting. 
(2) We propose DiffusEmp, a novel diffusion model-based framework, to unify the utilization of dialogue context and control signals, achieve elaborate control with a specific masking strategy, and integrate an emotion-enhanced matching method to produce diverse responses for a given context.
(3) Experimental results show that our method outperforms competitive baselines in generating informative and empathetic responses.

## Setup
```bash 
pip install -r requirements.txt 
```

## Datasets
EmpatheticDialogue dataset comprises 24,850 open-domain multi-turn conversations between two interlocutors.
To download the EmpatheticDialogues dataset:
```bash 
wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
```

## Train
```bash
cd scripts
bash train.sh
```

## Decode
```bash
cd scripts
bash run_decode.sh
```

## Evaluation
```bash
cd scripts
python eval_seq2seq.py 
```
# DiffusEmp
