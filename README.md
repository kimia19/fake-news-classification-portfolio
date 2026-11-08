# Fake News Classification — FA-KES

Independent, research-oriented portfolio implementation of **title-based fake-news classification** using TF-IDF/Logistic Regression and frozen **GloVe 6B 100d** embeddings with LSTM, CNN–LSTM and regularized CNN–LSTM.

This is a clean publication edition, *not a redistribution of an original course notebook*. The original coursework source and its historical Git commits are intentionally excluded. The rewritten implementations are conceptually similar, but reproducibility of exact historical scores is **not guaranteed**.

## Data and provenance

FA-KES: Abu Salem et al. (2019), https://doi.org/10.5281/zenodo.2607278 (CC BY 4.0). Dataset file is obtained at execution time with an MD5 check. Its label convention is **0=fake, 1=credible**. The corpus is small (804 articles) and has repeated normalized headlines; grouped train/validation/test splitting prevents one form of exact-title leakage.

Embeddings: Pennington et al., GloVe (2014), https://nlp.stanford.edu/projects/glove/. Download is via checksum-verified binary repack https://doi.org/10.5281/zenodo.4925376. Pretrained vectors are not included in this repository.

## Reproduce validation-only experiments

Python 3.11 recommended.

```bash
python -m pip install -r requirements.txt
python download_fakes.py
python download_glove_100d.py
python reproduce_tfidf.py
python neural_experiment.py --seed 42 --epochs 10 --stem
```

The neural public-edition runner outputs **validation metrics only**, selects one architecture using validation macro-F1, and does **not** score the repeatedly inspected test fold. Text preprocessing in this edition is an explicit implementation choice and may differ from historical notebook preprocessing. Experiment results are therefore exploratory and cannot establish external generalization.

## Historical research context

A separate private coursework reproduction previously reported a validation-selected CNN–LSTM test macro-F1 of about **0.565** on the FA-KES split, with substantial seed sensitivity. That partition was already inspected in earlier experiments; **do not present this as a fresh blind evaluation or as a measured result of the independent code above**.

## Scope and limitations

Small dataset; title-only lexical input; incomplete domain-shift validation; fixed GloVe embeddings. The source code here is a fresh portfolio rewrite and contains no instructor-provided assignment notebook, private Colab metadata or training photos. Source corpus and vectors remain subject to their original licenses. No license is currently granted for redistribution of the independently authored source code beyond applicable platform terms; add an explicit code license only with owner approval.
