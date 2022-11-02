# Molecule generation

[GraphINVENT](https://github.com/MolecularAI/GraphINVENT) is used for molecule generation and [GEFA](https://github.com/ngminhtri0394/GEFA) is used for DTA prediction.

## Step 1: Generating molecules

Follow the instructions in the [GraphINVENT](../GraphINVENT/tutorials) tutorial to generate molecules.

## Step 2: Preparing the Data

* [Extract features from proteins](Extracting_features_from_proteins.md) and place them in the appropriate folder.

* Copy the generated SMILEs to the GEFA/data/{dataset}/smiles directory.

* Prepare the data by running the [prepare_data.py](../GEFA/prepare_data.py) script in the [GEFA](../GEFA) folder.

```bash
python prepare_data.py {dataset}
```

## Step 3: DTA prediction

* Set the appropriate values in [config.py](../GEFA/config.py).

* Run the DTA prediction.

```bash
python predict.py
```
