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
```bash
cp XMC/llms4subjecs-xmc-arctic-epoch-3/final/augmenter_state.bin XMC/GND-Subject-test-arctic_m_v2/final
python eval-rrf.py --model_path /media/4t/FireFox-Download/test/RAF-SI-main/XMC/GND-Subject-test-arctic_m_v2/final --dataset_dir Datasets/llms4subjecs-xmc --reranker_path Results/rerank/llms4subjecs-xmc/model/reranker-final.pth --rrf_k 3
```
