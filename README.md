# CDRMamba: A Framework for Automated CranioMaxilloFacial Defect Reconstruction Using Mamba-Based Modeling
This repository contains the code for the paper "CDRMamba: A Framework for Automated CranioMaxilloFacial Defect Reconstruction Using Mamba-Based Modeling".
## Directory Structure


```
.
├── config.py
├── dataset
│   ├── dataset_lits_train.py
│   ├── dataset_lits_val.py
│   └── transforms.py
├── models
│   ├── __init__.py
│   ├── CDRMamba.py
│   └── TSSMamba.py
├── train.py
├── test.py
└── utils
    ├── common.py
    ├── logger.py
    ├── loss.py
    ├── metrics.py
    └── weights_init.py
```

## Dependencies

Please ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch
- tqdm
- SimpleITK
- numpy
You can set up the conda environment using the `requirements.txt` file:

```bash
conda create --name myenv --file requirements.txt
conda activate myenv
bash```


## Usage
### Training the Model

You can train the model using the following command:

```bash
python train.py
```

Test command:

```bash
python test.py
```


### Configuration

You can set the training parameters such as learning rate, batch size, etc., in the [`config.py`](vscode-file://vscode-app/d:/Users/%E6%9B%BE%E5%AE%BD%E4%B8%80/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) file.

## Weights and Dataset

The weights and dataset are open and can be obtained by contacting [22210860059@m.fudan.edu.cn](vscode-file://vscode-app/d:/Users/%E6%9B%BE%E5%AE%BD%E4%B8%80/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
