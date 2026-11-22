# Skin Lesion Classification

Binary classifier distinguishing malignant from benign skin lesions on the
ISIC 2018 archive, built in PyTorch. MSc Data Science dissertation, Manchester
Metropolitan University.

The interesting part of this problem is not the architecture. It is that only
about 20% of cases are malignant, which makes the obvious metric useless and
makes the choice of what to optimise the real decision.

---

## The problem with accuracy here

The test set holds 1,205 benign and 307 malignant images. A model that predicts
"benign" for every single image scores **79.7% accuracy** and finds no cancer
at all.

Every model in this project scores below that number, deliberately. Accuracy is
bought by ignoring the minority class, and the minority class is the entire
point. The measures that matter are recall on malignant cases and ROC-AUC.

---

## Data

ISIC 2018, using the official Training / Validation / Test releases rather than
a self-made split, so results stay comparable with published work and nothing
leaks between sets.

| Split | Images | Benign | Malignant | Malignant share |
|---|---:|---:|---:|---:|
| Train | 10,015 | 8,061 | 1,954 | 19.5% |
| Validation | 193 | 149 | 44 | 22.8% |
| Test | 1,512 | 1,205 | 307 | 20.3% |

---

## Method

**Class imbalance** handled with weighted binary cross-entropy,
`pos_weight = 4.13`, so a missed malignant case costs roughly four times a
false alarm. That ratio comes from the training class balance, not from tuning.

**Three models compared** on identical data and identical evaluation:

- a baseline CNN trained from scratch, as a floor
- EfficientNetB0, transfer learning
- DenseNet201, transfer learning

**Two-stage fine-tuning** for the pretrained models: train the classifier head
with the backbone frozen, then unfreeze the upper blocks and continue at a
reduced learning rate.

**Threshold tuning** after training. The default 0.5 cut-off is arbitrary when
the classes are imbalanced and the errors cost different amounts, so the
operating point was selected on the validation set rather than assumed.

---

## Results

Test set, 1,512 images. Arrows show the change from the default 0.5 threshold
to the tuned operating point.

| Model | Accuracy | Precision (mal.) | Recall (mal.) | F1 (mal.) | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 0.622 → 0.672 | 0.316 → 0.347 | 0.736 → 0.697 | 0.442 → 0.463 | 0.730 → 0.761 |
| EfficientNetB0 | 0.748 → 0.751 | 0.439 → 0.442 | 0.870 → **0.873** | 0.584 → 0.587 | **0.880** → 0.879 |
| DenseNet201 | 0.688 → 0.706 | 0.386 → 0.400 | 0.906 → 0.893 | 0.541 → 0.552 | 0.873 → 0.872 |

**EfficientNetB0 catches 268 of 307 malignant cases** at 0.88 ROC-AUC.
DenseNet201 catches slightly more, 274, but raises more false alarms to do it.
Which of those is the better model depends entirely on what a missed cancer
costs relative to an unnecessary referral, which is a clinical question rather
than a modelling one.

Transfer learning is worth roughly 0.12 ROC-AUC over the scratch-built CNN,
which is the clearest single finding here.

---

## What I got wrong

I selected the best training epoch on validation **accuracy** while using
weighted loss to fight the imbalance. Those two things work against each other:
the loss function was told to care about the rare class, and then the model
selection quietly preferred whichever epoch ignored it.

Switching the selection criterion to recall on the malignant class fixed it.
The lesson is that the metric has to be chosen before training, from the cost
of each kind of error, and it has to be applied consistently everywhere the
model is scored.

---

## Limitations

- The validation set holds only **44 malignant images**, so threshold selection
  is coarse and the chosen operating point carries real uncertainty.
- ISIC is a curated dermoscopic archive and is not representative of the images
  a general practice would produce.
- No external validation on a second dataset.
- Nothing here is clinically validated and none of it is a diagnostic tool.

---

## Repository

```
notebooks/   skin_lesion_pytorch.ipynb   full pipeline, training and evaluation
figures/     confusion matrices, ROC curves, training curves
README.md
```

## Running it

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib
```

Download the ISIC 2018 Task 3 archive from the
[ISIC Challenge site](https://challenge.isic-archive.com/data/) and point the
data path in the notebook at it. The dataset is not included here.

---

Built with PyTorch, Pandas, NumPy, Scikit-learn and Matplotlib.
