# Random Forest Classifier

Inspired by https://github.com/sile/randomforest

## Run

Simply add the files `train.csv` and `evaluate.csv` to the `./data` folder, and run `./train.sh`.

The resulting labels will be created at `./out/evaluated_labels.txt`.

## Details

This is a random forest classifier, the settings that work best for this data I have found to be `n_trees: 200`, `max_depth: 12`, `bagging_percentage: 50%`.

It reaches a consistent ~59.2% test classification accuracy on a training/testing split of 70%.