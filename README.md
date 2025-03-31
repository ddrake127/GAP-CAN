# GAP-CAN
This repository contains the source code for the GAP-CAN framework, which searches for adversarial samples in CAN Bus message streams.

## Citations
#### If you use the software presented in this repository, please cite as:

Devin Drake, Victor Cobilean, Harindra Mavikumbure, Morgan Stuart,
Swagat Das, and Milos Manic. Generating adversarial samples for
transformer-based can bus anomaly detectors. In ICPS 2025 – 8th IEEE
International Conference on Industrial Cyber-Physical Systems, pages 1–
7. IEEE, May 2025.

#### Furthermore, if you utilize the datasets from this repository, please cite them as well:

CIC Dataset:
Euclides Carlos Pinto Neto, Hamideh Taslimasa, Sajjad Dadkhah,
Shahrear Iqbal, Pulei Xiong, Taufiq Rahman, and Ali A Ghorbani.
Ciciov2024: Advancing realistic ids approaches against dos and spoofing
attack in iov can bus. Internet of Things, 26:101209, 2024.

HCRL Dataset:

Hyun Min Song, Jiyoung Woo, and Huy Kang Kim. In-vehicle network
intrusion detection using deep convolutional neural network. Vehicular
Communications, 21:100198, 2020.

## Getting Started

1. `pip install torch pandas numpy`
2. `python3 gumbel_softmax_tester.py` to run the space search on both datasets.
