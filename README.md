# RAF-SI

## dataset

llms4subjects: https://github.com/jd-coderepos/llms4subjects/tree/main/shared-task-datasets

EURLex-4k: see https://colab.research.google.com/github/nilesh2797/DEXML/blob/main/dexml.ipynb

llms4subjects/EURLex-4k is format: see https://sandbox.zenodo.org/records/504982

```latex  
  lbl.json
  trn.json
  tst.json
```

### requirements
```bash
pip install -r requirements.txt
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python setup.py install

```

### llms4subjects train
```bash
python train-sbert.py
python train-sbert-aug.py
python train-sbert-rerank.py --model_path XMC/GND-Subject-test-arctic_m_v2/final --dataset_dir Datasets/llms4subjecs-xmc
```


### llms4subjects eval

python eval-rrf.py --model_path /media/4t/2026/elmo-main/XMC/GND-Subject-test-arctic_m_v2-epoch-3/final --dataset_dir Datasets/GND-Subject-test --reranker_path Results/rerank/GND-Subject-test/model/reranker-final.pth --rrf_k 3
