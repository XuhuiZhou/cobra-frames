# Cobra Frames
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3109/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

## Getting started
* The source code is built aroud PyTorch, and has the following main dependencies:

    - Python 3.8
    - PyTorch 1.12.1
    - transformers 4.22.2

    For more extensive dependencies, see `requirements.txt`.
    A recommended way to install the dependencies is via [Anaconda](https://www.anaconda.com/download/):
    ```bash
    conda create -n cobra python=3.8
    conda activate cobra
    conda install -c conda-forge pip # make sure pip is installed
    python -m pip install -r requirements.txt # make sure the packages are installed in the specific conda environment
    python -m pip install -e .
    ```

* We use `pre-commit` hooks to unify the code style, please refer to https://pre-commit.com/ for installation and usage.

## Training and Evaluation
Please refer to `scripts/explain_model/train_explain_model.sh` and `scripts/predict_explain_model.sh` for training and evaluation scripts.

## Citation

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{zhou2023cobraframes,
  title={COBRA Frames: Contextual Reasoning about Effects and Harms of Offensive Statements},
  author={Zhou, Xuhui and Zhu, Hao and Yerukola, Akhila and Davidson, Thomas and Hwang, Jena D. and Swayamdipta, Swabha and Sap, Maarten},
  year={2023},
  booktitle={Findings of ACL},
  url={http://arxiv.org/abs/2306.01985}
}
```




