# Deep Learning Speech Recognition

## 🚀 Introduction

In this project, we evaluated and compared the performance of three deep learning architectures -- the M5 CNN, Vision Transformer (ViT), and Conformer -- on a multi-class speech command classification task. We investigated the impact of key model hyperparameters, audio input representations, and dataset balancing techniques on classification accuracy. Based on our experimental results, we developed and proposed an ensemble approach combining multiple M5 CNN classifiers to enhance overall performance.

## 📂 Folder Structure

```plaintext
📦deep-learning-speech-recognition
 ├── 📂configs                # Configuration files for experiments
 │   ├── 📄config_utils.py    # Utils for showing or saving configs
 │   └── 📄config.py          # Main configuration script
 ├── 📂configuration          # Experiment-specific configuration files
 │   ├── 📂10_class_m5_sweep
 │   ├── 📂10_class_vit_repr
 │   ├── 📂10_class_vit_sweep
 │   ├── 📂binary_test
 │   ├── 📂conformer_scratch_size_config
 │   ├── 📂full_dataset_sweep
 │   ├── 📂multiclass_test
 │   ├── 📂run_optimal_configs
 │   ├── 📂sampling_strategy
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