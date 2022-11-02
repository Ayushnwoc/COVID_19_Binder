# Setting Up Environment

Set up the environment for extracting features from proteins, molecule generation and DTA prediction.

## Extracting Features From Proteins

### **TAPE**

TAPE is installed as a python package. It can be installed using either the instructions given in [TAPE](https://github.com/songlab-cal/tape)'s README or by running the following command:

```bash
pip install tape-proteins
```

Make sure that you install this on a separate environment.

### **RaptorX**

Setup RaptorX in a separate environment according to the instructions given on the [RaptorX-3DModeling](https://github.com/j3xugit/RaptorX-3DModeling) repository.

* Install the required external packages for all modules and the required packages for contact/distance/orientation/angle prediction.

* Install HHblits for MSA generation. We have used hhsuite-3.3.0 and the UniRef30_2020_06 database for MSA generation.

* Install deep learning models for contact/distance/orientation/angle/SS/ACC prediction.

* Setup environment variables.

Only the HHDIR and HHDB environment variables need to be set in the raptorx-external.sh script, we won't need the others.

### **Predict Property**

Follow the instructions in [Predict_Property](https://github.com/realbigws/Predict_Property)'s README. A separate environment is not required.

## Molecule Generation and DTA Prediction

### **Python**

Python version 3.8 was used.

### **PyTorch**

PyTorch version 1.9.0, with CUDA 11.1 was used. The CUDA version for PyTorch should match your CUDA version.

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Installation instructions: <https://pytorch.org/get-started/previous-versions/>

### **PyTorch Geometric**

Pytorch Geometric version compatible with the installed PyTorch and CUDA versions should be installed.

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric
```

 Installation instructions: <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>

### **RDKit**

RDKit 2021.03.4 was used.

```bash
conda install -c conda-forge rdkit
```

Installation instructions: <https://www.rdkit.org/docs/Install.html>