Grand Challenge: RAVIR
==============

# About

This repository contains our contribution to the [RAVIR Grand Challenge](https://ravir.grand-challenge.org/RAVIR/) rolling competition. It's an image segmentation challenge where participants develop some form of machine learning model to identify arteries and veins in a set of images. Images containing the model predictions is then uploaded to the RAVIR website for automatic evaluation using both Dice and Jaccard scores, contributions are then ranked on a leaderboard depending on their combined score. RAVIR is challenging due to the small dataset combined with small structures, requiring clever solutions to both problems.

I were one of two members in the [KTH-CBH-CM2003-1](https://ravir.grand-challenge.org/teams/t/2914/) team. We developed a deep learning model using a U-Net architecture, an architecture that have displayed high performance in medical image segmentation tasks, and were able to achieve a combined score of `0.6026 ± 0.1222`, placing us [41st](https://ravir.grand-challenge.org/evaluation/96742895-eae3-4614-8af7-655f4bd7e2a3/) on the RAVIR leaderboard at the time of submission.

If you're interested and want to learn more, see `Report.md` in this repository.

# Running instructions

Unfortunetly, model weights were not saved for posterity. If you're interested in the model you will have to train it yourself:

1. Download the dataset from the [RAVIR Grand Challenge](https://ravir.grand-challenge.org/RAVIR/) website.
2. Install [Jupyter Notebook](https://jupyter.org/install).
3. Download the `code` directory and import it to your Jupyter Notebook instance.
4. Open `prerequisite.ipynb` to read about the prerequisites and install dependencies.
5. In `pipeline.ipynb` and `utils.py`, change directory paths to accomondate your own environment.
6. Run one of the six pipelines.
7. View model predictions in the `predictions` directory.

# Experience Gained

Practical deep learning methods and tools.