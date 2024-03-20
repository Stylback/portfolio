RAVIR Challenge
==============

**Authors:** *JONAS STYLBÄCK, MIRANDA GISUDDEN*

# Table of contents

- [RAVIR Challenge](#ravir-challenge)
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Methods](#methods)
    - [Table 1: Baseline hyperparameters](#table-1-baseline-hyperparameters)
- [Results](#results)
  - [Initial evaluation](#initial-evaluation)
  - [Hyperparameter tuning (configuration 4)](#hyperparameter-tuning-configuration-4)
- [Discussion and further improvements](#discussion-and-further-improvements)
- [References](#references)

# Introduction

The microvasculature system plays a role in diseases such as diabetes. This system can be directly observed only in the retina, which makes this area interesting for research. The RAVIR dataset<sup>[1][2]</sup> is a new dataset that can be used for segmentation of veins and arteries in the retina. In this project, a deep learning model was developed for segmentation of retinal veins and arteries.

# Methods

The model used is of UNET-type, which is a neural network that was developed specifically for medical image segmentation. We tested six different model configurations, three for binary classification and another three for multiclass classification. Multiclass configurations were also implemented with data augmentation and K-fold cross-validation.

The training set consist of 23 images of retinal vessels each with a corresponding mask while the test set contains 19 images of retinal vessels with no corresponding masks. The training set was split into training and validation subsets with a validation ratio of 30%.

Preliminary evaluation of model performance was done using dice coefficient, after which prediction results was uploaded to the RAVIR Challenge website for server-sided evaluation. 

Hyperparameters stayed the same during initial evalutation of each configuration ([table 1](#table-1-baseline-hyperparameters)), with hyperparameter tuning being performed only on the most promising configuration.

### Table 1: Baseline hyperparameters

| Hyperparameter | Value | Comment |
| --- | --- | --- |
| Image dimensions (W, H, D) | 768, 768, 1 | Original input size |
| Batch size | 1 | With such a small dataset, optimizing for training time is not needed |
| Validation ratio | 0.3 | Provides 8 validation samples |
| Number of filters in base layer | 8 | Good starting point |
| Optimizer | Adam | Standard optimizer, efficient and well suited for segmentation tasks<sup>[3]</sup> |
| Learning rate | 1e-4 | Good middle-ground for initial evaluation |
| Epochs | 150 | Good middle-ground for initial evaluation |
| Metric | Dice coefficient | Standard metric for segmentation tasks, also used during server-sided evaluation |
| Dropout rate | 0.2 | Only used in configuration 2, 3, 5 and 6 |
| Number of folds | 3 | Only used in configuration 4, 5 and 6 |

# Results

## Initial evaluation

| Configuration | Performance histogram | Prediction of `IR_Case_006` | Mean Dice score |
| --- | --- | --- | --- |
| Configuration 1: Binary classification with standard UNET | ![histogram_1](/ravir-challenge/media/histogram_1.png) | ![prediction_1](/ravir-challenge/media/prediction_1.png) | 0.3176 ± 0.0567 |
| Configuration 2: Binary classification with Dropout | ![histogram_2](/ravir-challenge/media/histogram_2.png) | ![prediction_2](/ravir-challenge/media/prediction_2.png) | 0.3281 ± 0.0830 |
| Configuration 3: Binary classification with Dropout and Batch Normalization | ![histogram_3](/ravir-challenge/media/histogram_3.png) | ![prediction_3](/ravir-challenge/media/prediction_3.png) | 0.2653 ± 0.0990 |
| Configuration 4: Multiclass classification with data augmentation and K-fold cross-validation | ![histogram_4](/ravir-challenge/media/histogram_4.png) | ![prediction_4](/ravir-challenge/media/prediction_4.png) | **0.4694 ± 0.1137** |
| Configuration 5: Multiclass classification with data augmentation, K-fold cross-validation and Dropout | ![histogram_5](/ravir-challenge/media/histogram_5.png) | ![prediction_5](/ravir-challenge/media/prediction_5.png) | 0.4234 ± 0.1380 |
| Configuration 6: Multiclass classification with data augmentation, K-fold cross-validation, Dropout and Batch Normalization | ![histogram_6](/ravir-challenge/media/histogram_6.png) | ![prediction_6](/ravir-challenge/media/prediction_6.png) | 0.3471 ± 0.1751 |

## Hyperparameter tuning (configuration 4)

| Configuration | Performance histogram | Prediction of `IR_Case_006` | Mean Dice score |
| --- | --- | --- | --- |
| `base=8`, `learning_rate=1e-4`, `epochs=300` | ![histogram_7](/ravir-challenge/media/histogram_7.png) | ![prediction_7](/ravir-challenge/media/prediction_7.png) | 0.4987 ± 0.1100 |
| `base=16`, `learning_rate=1e-4`, `epochs=300` | ![histogram_8](/ravir-challenge/media/histogram_8.png) | ![prediction_8](/ravir-challenge/media/prediction_8.png) | 0.4832 ± 0.1687 |
| `base=16`, `learning_rate=1e-3`, `epochs=300` | ![histogram_9](/ravir-challenge/media/histogram_9.png) | ![prediction_9](/ravir-challenge/media/prediction_9.png) | **0.6026 ± 0.1222** |
| `base=32`, `learning_rate=1e-3`, `epochs=300` | ![histogram_10](/ravir-challenge/media/histogram_10.png) | ![prediction_10](/ravir-challenge/media/prediction_10.png) | 0.5855 ± 0.1328 |

# Discussion and further improvements

![leaderboards_placement](/ravir-challenge/media/leaderboards.png)

We are happy with the [results](https://ravir.grand-challenge.org/evaluation/96742895-eae3-4614-8af7-655f4bd7e2a3/), placing 41st on the RAVIR leaderboard with a mean dice score of `0.6026 ± 0.1222`, it was a welcoming bounce back from a disappointing result of `0.0169 ± 0.0119` [last november](https://ravir.grand-challenge.org/evaluation/66e6546d-1556-4596-a03e-bef55356c81a/).

We noticed during initial testing (configuration 1-3) that the models easily overfitted due to the small dataset, to combat this we implemented both data augmentation and K-fold cross-validation for the remaining three configurations.

We were surprised with the high performance of configuration 4, especially after tuning the hyperparameters. This stands in contrast to the configurations utilizing Dropout and Batch Normalization (_2, 3, 5 and 6_), from which we were hoping for an upswing in performance. We believe the small dataset and batch size did not give the right conditions for these configurations to have their intended impact.

While satisfied with our results, we believe there is still room for improvement. The task was challenging due to the limited dataset, small structures and high similarity between object classes. To climb the leaderboard we believe it would be necessary to increase the dataset, for example by dividing each image into sections of 4 or 9 and training the model on these subsections. Using weight-maps might also increase performance as it helps the model distinguish small structures from the background, it would however be necessary to completely rework both model and pipeline to implement this.

# References

[1]: Hatamizadeh, A., Hosseini, H., Patel, N., Choi, J., Pole, C., Hoeferlin, C., Schwartz, S. and Terzopoulos, D., 2022. [RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging](https://ieeexplore.ieee.org/abstract/document/9744459?casa_token=bVo3jBzkoxYAAAAA:t2O9G3Y6cK05AxFEBL8oi_PyrOzbypsBHNCiKuqkiYTNG3gwzy7oNVo3dPpSmxFS-9B2dZJJzmjP1Q). IEEE Journal of Biomedical and Health Informatics.

[2]: Hatamizadeh, A., 2020. [An Artificial Intelligence Framework for the Automated Segmentation and Quantitative Analysis of Retinal Vasculature. University of California](https://escholarship.org/uc/item/4r63v2bd), Los Angeles.

[3]: Kingma, D. P., & Ba, J. (2014). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)