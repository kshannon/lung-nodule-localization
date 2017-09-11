# ucsd-dse-capstone
Capstone Project for Data Science &amp; Engineering Master's Program at UCSD

Computed tomography (CT) has recently replaced conventional X-ray as the primary
screening tool for lung cancer because it has been shown to reduce mortality by as much as
20% in high risk patients (Aberle et al. , 2011). Unfortunately, there is a high false positive rate
(FPR) associated with CT screening. At least one “lung nodule” is detected in half of all CT
scans, but only 10% of these “nodules” are in fact cancerous.

The LUng Nodule Analysis (LUNA16) dataset was constructed to help improve and
compare the detection and classification algorithms used to assist radiologists in lung CT
screening. It consists of 888 CT scans of the lungs published in 2016 by the National Cancer
Institute under the Creative Commons Attribution 3.0 Unsupported License. The dataset is over
140 GB in size and includes over 750,000 regions of interest (ROI). A set of four expert
radiologists labeled 1,186 of these ROIs as true nodules (class 1) and the remainder as
non-nodules (class 0).

The goal of this project is to use machine learning methods, including but not limited to
convolutional neural networks, to improve both localization and classification of lung nodules
within this dataset. The LUNA16 website ( https://luna16.grand-challenge.org/home/ ) includes a
leaderboard of both industry and academic teams which have achieve the highest scores in one
of two metrics: (1) nodule detection and localization and (2) nodule false positive reduction.
