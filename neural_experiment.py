"""Independent portfolio rewrite of a title-based FA-KES neural experiment.

Architectures are described explicitly rather than importing copyrighted or
unknown-license coursework notebook cells. This public edition is a *new*
experimental implementation: its numerical outputs need not match the archived
coursework results, and must not be passed off as a fresh blind benchmark.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from download_fakes import EXPECTED_MD5
from audit_fakes import audit_dataset
from glove_binary import build_glove_matrix
from split_fakes import make_group_split


def clean(text, stem=False):
    tokens=re.findall(r"[a-z]+",str(text).lower())
    if stem:
        from nltk.stem import PorterStemmer
        porter=PorterStemmer()
        tokens=[porter.stem(w) for w in tokens]
    return " ".join(tokens)


def make_network(kind, matrix, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    embed=layers.Embedding(matrix.shape[0],matrix.shape[1],
        weights=[matrix],trainable=False)
    stack=[layers.Input((40,)),embed]
    if kind=="LSTM":
        stack += [layers.LSTM(32)]
    elif kind=="CNN-LSTM":
        stack += [layers.Conv1D(128,5,activation="relu"),
                  layers.MaxPooling1D(2),layers.LSTM(32)]
    elif kind=="Regularized CNN-LSTM":
        stack += [layers.SpatialDropout1D(0.2),
                  layers.Conv1D(128,5,activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                  layers.MaxPooling1D(2),
                  layers.LSTM(32,dropout=0.3,recurrent_dropout=0.3),
                  layers.Dropout(0.3)]
    else:
        raise ValueError("Unsupported model "+kind)
    stack.append(layers.Dense(1,activation="sigmoid"))
    model=Sequential(stack)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-4 if kind.startswith("Regularized") else 1e-3),
        loss="binary_crossentropy",metrics=["accuracy"])
    return model


def run(data,glove,output,seed=42,epochs=10,stem=False):
    meta=audit_dataset(data)
    if meta["md5"]!=EXPECTED_MD5:
        raise ValueError("Only checksum-verified FA-KES files are accepted")
    frame=pd.read_csv(data,encoding="latin1",
        usecols=["article_title","labels"],keep_default_na=False)
    titles=frame["article_title"].astype(str).to_numpy()
    targets=frame["labels"].astype(int).to_numpy()
    train,val,test,_=make_group_split(titles,targets,seed)
    prepared=[clean(t,stem) for t in titles]
    tokenizer=Tokenizer(num_words=5000,oov_token="<OOV>")
    tokenizer.fit_on_texts([prepared[i] for i in train])
    matrix,coverage=build_glove_matrix(glove,tokenizer.word_index,
        vocab_size=5000,embedding_dim=100)
    ids=pad_sequences(tokenizer.texts_to_sequences(prepared),
        maxlen=40,padding="post",truncating="post")
    records={}
    candidates={}
    for kind in ("LSTM","CNN-LSTM","Regularized CNN-LSTM"):
        model=make_network(kind,matrix,seed)
        history=model.fit(ids[train],targets[train],
            validation_data=(ids[val],targets[val]),verbose=0,
            epochs=epochs,batch_size=64,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",patience=3,restore_best_weights=True)]
                if kind.startswith("Regularized") else [])
        pred=(model.predict(ids[val],verbose=0).ravel()>=0.5).astype(int)
        val_f1=float(f1_score(targets[val],pred,average="macro",zero_division=0))
        records[kind]={"validation_macro_f1":val_f1,
            "training_loss":[float(x) for x in history.history["loss"]],
            "validation_loss":[float(x) for x in history.history["val_loss"]]}
        candidates[kind]=model
        print(kind,"validation macro-F1",round(val_f1,4),flush=True)
    winner=max(records,key=lambda k:records[k]["validation_macro_f1"])
    report={
        "study":"FA-KES portfolio implementation, frozen pretrained GloVe",
        "qualification":"Exploratory educational experiment. Test split previously inspected in precursor coursework; not newly blind.",
        "seed":seed,"epochs_max":epochs,"stemming":stem,
        "dataset_md5":meta["md5"],"glove":coverage,
        "partition_counts":{"train":len(train),"validation":len(val),"test_not_scored":len(test)},
        "validation_results":records,"selected_architecture":winner,
        "test_scored":False,
        "limitations":["Small historical data","Only lexical titles","Previously examined source dataset","Architecture code rewritten for public release"]
    }
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=Path("data/FA-KES-Dataset.csv"))
    parser.add_argument("--glove",type=Path,default=Path("data/glove.6B.100d.zip"))
    parser.add_argument("--output",type=Path,default=Path("outputs/fakes_portfolio_validation.json"))
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--stem",action="store_true")
    args=parser.parse_args()
    run(args.data,args.glove,args.output,args.seed,args.epochs,args.stem)
