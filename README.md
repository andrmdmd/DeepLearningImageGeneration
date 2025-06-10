# Deep Learning Image Generation

## 🚀 Introduction

In this project, we evaluated and compared the performance of multiple deep learning architectures on image generation tasks. We investigated the impact of key model hyperparameters, data augmentation techniques, and dataset balancing strategies on generation quality. Based on our experimental results, we developed and proposed adaptive augmentation methods to enhance model performance.

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
 ├── 📂adaptive_augmentation  # Adaptive augmentation methods
 │   ├── 📄adaptive_augment.py
 │   ├── 📄adaptive_augmentation.py
 ├── 📂charts                 # Visualization outputs
 │   ├── 📄training_class_distribution.png
 │   ├── 📄unknown_method_validation_acc.png
 │   └── 📂final_confusion_matrices
 ├── 📂configs                # Configuration files for experiments
 │   ├── 📄config_utils.py    # Utils for showing or saving configs
 │   └── 📄config.py          # Main configuration script
 ├── 📂configuration          # Experiment-specific configuration files
 ├── 📂dataset                # Data loading and preprocessing modules
 ├── 📂engine                 # Training and validation engine
 ├── 📂logs                   # Logs for experiments
 ├── 📂modeling               # Model architecture definitions
 ├── 📂notebooks              # Jupyter notebooks for analysis
 ├── 📂utils                  # Utility scripts for various tasks
 ├── 📂wandb                  # WandB experiment tracking
 ├── 📄.gitignore             # Specifies intentionally untracked files
 ├── 📄LICENSE                # License file
 ├── 📄README.md              # Project README
 ├── 📄linter.sh              # Code formatting script
 ├── 📄main.py                # Main training script
 ├── 📄pyproject.toml         # Project dependencies and settings
 ├── 📄run_sweeps.sh          # Script to run WandB sweeps
 ├── 📄sweep.py               # Sweep training script
 └── 📄requirements.txt       # Dependencies
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

    Example for training a model:

    ```shell
    python3 main.py --config configuration/dcgan_data_augmentation_sweep/config.json
    ```

3.  **Run WandB sweeps:**

    Example to run a WandB sweep for data augmentation experiments:

    ```shell
    ./run_sweeps.sh
    ```

    Ensure you are logged into your WandB account.