<h1 align="center">Panini: a transformer-based grammatical error correction method for Bangla</h1>
<p align="center">
   <a href="https://doi.org/10.1007/s00521-023-09211-7" target="_blank">Paper @ Neural Computing and Applications</a> and <a href="https://huggingface.co/datasets/nahid-hub/BanglaGEC" target="_blank">Dataset @ HuggingFace</a>
</p>

## Get Started

```
git clone https://github.com/mehedihasanbijoy/BanglaGEC.git
```

or manually **download** and **extract** the github repository of BanglaGEC.

## Environment Setup

### Create A Virtual Environment

```
conda env create -f environment.yml
```

### Activate the Environment

```
conda activate BanglaGEC
```

## Prepare the BGEC Corpus

```
python -c "from datasets import load_dataset; load_dataset('nahid-hub/BanglaGEC').save_to_disk('./Dataset')"
```

or manually **download** the corpus from <a href="https://huggingface.co/datasets/nahid-hub/BanglaGEC" target="_blank">here</a> and keep the extracted files into **./Dataset/**

## Training and Evaluation

### Panini

```
python main.py --CORPUS_PATH "./Dataset/corpus.csv" --KNOWLEDGE_PATH "./KnowledgeToBeTransferred/paraphrase.pth" --CHECKPOINT_PATH "./Checkpoints/panini.pth" --MODEL_NAME "panini" --BATCH_SIZE 16 --N_EPOCHS 50
```

### BanglaT5

```
python main.py --CORPUS_PATH "./Dataset/corpus.csv" --CHECKPOINT_PATH "./Checkpoints/banglat5.pth" --MODEL_NAME "banglat5" --BATCH_SIZE 16 --N_EPOCHS 50
```

### T5-Small

```
python main.py --CORPUS_PATH "./Dataset/corpus.csv" --CHECKPOINT_PATH "./Checkpoints/t5small.pth" --MODEL_NAME "t5small" --BATCH_SIZE 16 --N_EPOCHS 50
```

## BibTeX Entry and Citation Info

```
@article{hossain2024panini,
  title={Panini: a transformer-based grammatical error correction method for bangla},
  author={Hossain, Nahid and Bijoy, Mehedi Hasan and Islam, Salekul and Shatabda, Swakkhar},
  journal={Neural Computing and Applications},
  volume={36},
  number={7},
  pages={3463--3477},
  year={2024},
  publisher={Springer}
}
```
