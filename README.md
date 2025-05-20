# Deep Learning Speech Recognition

## 🚀 Introduction

In this project, we evaluated and compared the performance of three deep learning architectures -- the M5 CNN, Vision Transformer (ViT), and Conformer -- on a multi-class speech command classification task. We investigated the impact of key model hyperparameters, audio input representations, and dataset balancing techniques on classification accuracy. Based on our experimental results, we developed and proposed an ensemble approach combining multiple M5 CNN classifiers to enhance overall performance.

## Adding dataset

1. Download ZIP from https://www.kaggle.com/datasets/borhanitrash/cat-dataset?resource=download
2. Unpack the ZIP
3. Rename the `cats` folder to `data`
4. Rename the `Data` folder to `cats`
5. Move `data` folder to the root of project directory

Structure change `cats/Data/` > `data/cats/`

## 📂 Folder Structure

```plaintext
📦deep-learning-image-generation
 ├── 📂configs                # Configuration files for experiments
 │   ├── 📄config_utils.py    # Utils for showing or saving configs
 │   └── 📄config.py          # Main configuration script
 ├── 📂configuration          # Experiment-specific configuration files
 │   └── 📂sweep_test
 ├── 📂dataset                # Data loading and preprocessing modules
 │   └── 📄dataset.py         # Data loader and preprocessing scripts
 ├── 📂modeling               # Model architecture definitions
 │   ├── 📄loss.py            # Loss function
 │   └── 📄model.py           # All architecture classes
 ├── 📂utils                  # Utility scripts for various tasks
 │   └── 📄metrics.py         # Performance metrics
 ├── 📂engine                 # Training and validation engine
 │   ├── 📄base_engine.py     # Base engine class
 │   ├── 📄sweep_engine.py    # Sweep engine class
 │   └── 📄engine.py          # Training and validation loops
 ├── 📄.gitignore             # Specifies intentionally untracked files
 ├── 📄LICENSE                # License file
 ├── 📄README.md              # Project README
 ├── 📄linter.sh              # Code formatting script
 ├── 📄requirements.txt       # Dependencies
 ├── 📄main.py                # Main training script
 └── 📄sweep.py               # Sweep training script
```

## ⚙️ Configuration

Experiment configurations are stored in the `configuration` directory. WandB sweeps are heavily utilized for hyperparameter tuning and experiment tracking. Reproducibility is maintained by setting random seeds in configuration files.

## 🏋️‍♂️ Training

### Basic Usage

1.  **Set up the environment:**

    ```shell
    pip install uv
    uv sync
    ```

2.  **Run training scripts:**

    Example for training a Ensemble model:

    ```shell
    python3 sweep.py --config configuration/full_dataset_sweep/config_11_classes.json --model.architecture="M5" \
    ```

3.  **Run WandB sweeps:**

    Example to run a WandB sweep for data augmentation experiments:

    ```shell
    ./configuration/data_augmentation/run_sweeps.sh
    ```

    Ensure you are logged into your WandB account.