# Geneformer-Distilled

This is a part of our seminar at the Institute of Statistics at LMU Munich. 

Here we are trying to distill a 4M parameter model from 10M parameter model which is the Geneformer, considered one of the seminal work in network biology. We have managed to compress the model from 10M to 4M around 2.5x reduction with around 1/25th of the data and 1/100th of the training with with identical things. It was a bit challenging to replicate this paper because there were a lot of things missing.

## Challenges Replicating Geneformer

1. Even to download files from Hugging Face is a mess.
2. Needed to create a custom data collator due to variable length sampling $\to$ to save computing.
3. Due to variable sequence length failed to use `torch.compile()` which builds a dynamic CUDA graph. 
4. Compute challenges, the data is massive with around 27 Mn rows and 500 tokens per sequences.  
5. Data was pretokenized that helped but also it was the most important part of the paper, needed to build our own collator. 
6. General Clarity Needed on what is the BERT masking strategy. They used the default by they should have mentioned. 
7. The dataset for the V2 models are not being provided 104 M.
8. Working with a 27 Mn rows with 500 tokens amounts to 10B tokens approx, really difficult to work with it. Major optimizations in dataset.py.
9. Problem of the unavailability of the 104Mn rows dataset so we couldn't validate the V2 & V3 version of the experiment. 

## Project Structure

- **src/**: Contains all source code for the project (models, trainer, data loaders, etc.).
- **configs/**: Configuration files for Hydra. `config.yaml` contains all hyperparameters and paths.
- **mainData/**: Directory storage for datasets.
- **outputs/**: Contains training logs and saved checkpoints.
- **notebooks/**: Jupyter notebooks for experiments and visualizations.

## How to Run

### 1. Data Preparation
Ensure your datasets are located in `mainData/`. The project expects Arrow format datasets.
To split the dataset into training and validation sets, run:
```bash
python -m src.data.split
```

### 2. Configuration
All experiment configurations are managed via Hydra in `configs/config.yaml`. You can modify model size, learning rates, batch sizes, and paths there.

### 3. Training
To start training the distillation process, use the `src.main` module.
For long running training sessions, it is recommended to use `nohup`:

```bash
nohup python -m src.main > training.log 2>&1 &
```
You can monitor the training progress by tailing the log file: `tail -f training.log` or checking the logs in `outputs/logs`.

### 4. Evaluation
To run quick evaluations on the trained models (Perplexity and MLM Accuracy):

```bash
python3 -m src.evals.quick
```

## Outputs and Checkpoints

The training artifacts are saved in the `outputs/` directory.
We have trained models of various sizes. You can find their checkpoints in specific directories:

- **4.3M Parameters**: `outputs/checkpoints_/geneformer4.3M/`
- **3M Parameters**: `outputs/checkpoints_/geneformer3M/`
- **2M Parameters**: `outputs/checkpoints_/geneformer2M/`

## Results

In general it proved difficult to replicate but not impossible. 
We have the following training metric for the **4.5M Model**:

![Training](notebooks/training_metrics_4.5M.png) 

The weights of the distilled models can be found in the outputs/checkpoints. We are going to use `model_best.pt`.

### Metrics Reference

1. **Accuracy**: Of the masking of the tokens.
2. **Perplexity**: Measure of how confident the model is while predicting the masked tokens.

Current numbers suggest that everything is working. If we train it more then the accuracy improves over time but the marginal gains are less. We increased the training steps from 61000 to 91000, we saw an improvement of 2% of MLM accuracy and perplexity improved by 40%.

```bash
============================================================
Metric          | Teacher (Target)   | Student (Yours)    | Gap       
MLM Accuracy    | 0.3059             | 0.2534             | -0.0524
Perplexity      | 15.40              | 22.48              | +7.08
============================================================
```

#### Results for Geneformer 3M

Currently training the 3M geneformer model with the minor config difference of `hidden_size` or `d_model` being reduced from 128 to 96.

![Training](notebooks/training_metrics_3M.png) 

```bash
============================================================
Metric          | Teacher (Target)   | Student (Yours)    | Gap       
MLM Accuracy    | 0.3028             | 0.1864             | -0.1164
Perplexity      | 15.59              | 38.91              | +23.32
============================================================
```

**Training observations**:
- **Data Scaling**: Training on more data improves MLM Accuracy and perplexity.
- **Double Descent**: observed around 150,000 steps where the model started converging more rapidly again.
- **Saturation**: After increasing steps from 300,000 to 600,000, only a minor bump in accuracy (~1%) was observed.
