#!/bin/bash

echo "[Master] Compiling" &&
cargo build --release &&

echo "[Master] Cleaning up ./out folder" &&
rm -rf ./out/* &&

echo "[Master] Preprocessing" &&
./preprocess.py encode_train ./data/train.csv ./out/train.dataset &&
./preprocess.py encode_evaluation ./data/train_nolabel.csv ./out/evaluate.dataset &&

echo "[Master] Fitting" &&
./target/release/challenge fit ./out/train.dataset ./out/classifier.serialized &&

echo "[Master] Evaluating" &&
./target/release/challenge evaluate ./out/evaluate.dataset ./out/classifier.serialized ./out/evaluated.txt &&

echo "[Master] Converting evaluated data to class labels" &&
./preprocess.py decode ./out/evaluated.txt ./out/evaluated_labels.txt &&

echo "[Master] Done"