# LSTM_on_Bias_Bios


## Fairness and Explainability in Gender Classification

This repository presents our research internship work on Fairness and Explainability in Machine Learning Models, where we detect and mitigate gender bias in text-based occupation classification using LSTM architectures and fairness-aware reweighting strategies. We also demonstrate the interpretability of our models using explainability frameworks (e.g., CERTIFAI in separate branches).

## Project Goals
````
- Train a gender classifier on the [Bias-in-Bios](https://arxiv.org/abs/1905.13372) dataset using a BiLSTM.
- Identify and mitigate gender bias in classification decisions.
- Apply **AIF360's Reweighing** algorithm for fairness preprocessing.
- Use **BERT** embeddings for textual representation.
- Train two models:
  - `lstm_gender_classifier.pth`: Baseline BiLSTM model.
  - `fair_lstm_gender_classifier.pth`: Fairness-aware LSTM using reweighted training.
- Visualize training metrics and confusion matrices (see notebooks).

---
````
## File Structure
````markdown
`lstm.py` - Baseline LSTM training on tabularized bios (no `hard_text`). 
`lstm_fair.ipynb` - Notebook analyzing fairness and accuracy of both models. 
`lstm_preprocess.ipynb` - Prepares, embeds, and visualizes the Bias-in-Bios data. 
`lstm_fairness.py` - Uses BERT + AIF360 reweighting to train a fair LSTM model. 
`lstm_gender_classifier.pth` - Trained baseline model weights. 
`fair_lstm_gender_classifier.pth` - Trained fairness-aware model weights. 
````

## Main libraries:

* `torch`
* `transformers`
* `aif360`
* `pandas`, `numpy`, `tqdm`
* `scikit-learn`

---

## How to Run

### Baseline LSTM

```bash
python lstm.py
```

### Fairness-Aware LSTM

```bash
python lstm_fairness.py
```

### Visualize and Compare

Open `lstm_fair.ipynb` to visualize:

* Accuracy vs fairness tradeoffs
* Confusion matrices
* Bias mitigation performance

---

## Evaluation

Metrics used:

* Accuracy
* Confusion Matrix
* Statistical Parity / Disparate Impact (via AIF360)
* Weighted Cross-Entropy (in fair model)

---

## References

* [AIF360 Toolkit](https://aif360.readthedocs.io/)
* [Bias in Bios Dataset](https://arxiv.org/abs/1905.13372)
* [BERT: Devlin et al. (2018)](https://arxiv.org/abs/1810.04805)

---

````
