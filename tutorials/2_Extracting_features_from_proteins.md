# Extracting features from proteins

[RaptorX-3DModeling](https://github.com/j3xugit/RaptorX-3DModeling) and [Predict_Property](https://github.com/realbigws/Predict_Property) repositories are used for extracting features from the proteins. The protein embeddings are extracted using [TAPE](https://github.com/songlab-cal/tape).

## Step 1: Protein embeddings generation

* Run the following command to generate embeddings from the protein sequence.

```bash
tape-embed transformer protein.fasta protein.npzcd  bert-base --tokenizer unirep --full_sequence_embed
```

* Store the npz file in the data/{dataset}/emb folder in the [GEFA](../GEFA/) folder.

## Step 2: Contact map prediction from protein sequence using RaptorX

* Use the BuildFeatures.sh script in the BuildFeatures folder for building the features from the protein sequence.

```bash
./BuildFeatures.sh protein.fasta
```

* Use the PredictPairRelation4OneProtein.sh script in the DL4DistancePrediction4/Scripts folder for generating the pair relations for one protein. The inputs will be the name of the protein and the contact folder generated in the previous step.

```bash
./PredictPairRelation4OneProtein.sh protein protein_OUT/protein_contact/
```

* Use the PrintContactPrediction.sh script in the DL4DistancePrediction4/Scripts folder for generating the contact map for the protein. The input will be the .predictedDistMatrix.pkl file from the previous step.

```bash
./PrintContactPrediction.sh protein.predictedDistMatrix.pkl
```

* Use the [txt2npy.py](../Utils/txt2npy.py) script in the [Utils](../Utils/) folder to convert the output.CM.txt file from the previous step to npy format. This will be contact map for the protein. Store this in the {data}/{dataset}/map folder in the [GEFA](../GEFA/) folder.

```bash
python txt2npy.py protein protein.CM.txt
```

## Step 3: Property Prediction from protein sequence

* Use the Predict_Property.sh script in the root folder of the Predict_Property repository for property prediction.

```bash
Predict_Property.sh -i protein.fasta
```

* Store the output folder in the data/{dataset}/profile/ folder in the [GEFA](../GEFA/) folder.
