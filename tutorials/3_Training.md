# Training

Train the models for generating molecules and DTA prediction.

## Generating Molecules

Follow the instructions in the [GraphINVENT](../GraphINVENT/tutorials) tutorial for training the model for molecule generation.

## DTA Prediction

* Navigate to the [GEFA](../GEFA) folder and fetch the davis dataset using git lfs.

```bash
git lfs fetch
```

* Extract the davis dataset.

```bash
tar -xf data.tar.gz
```

* Set the appropriate values in [config.py](../GEFA/config.py).

* Start training the model with 0 as 1 for argument. 0 if you want to start a fresh training process, 1 if you want to resume a training process.

```bash
python train_test.py 0
```
