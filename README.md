# GAP-CAN
This repository contains the source code for the GAP-CAN framework, which searches for adversarial samples in CAN Bus message streams.

## Citations
#### If you use the software presented in this repository, please cite as:

*Devin Drake, Victor Cobilean, Harindra Mavikumbure, Morgan Stuart, Swagat Das, and Milos Manic. GAP-CAN: Gradient-Based Adversarial Attack on Transformer-Based CAN Bus Anomaly Detectors. In ICPS 2025 – 8th IEEE International Conference on Industrial Cyber-Physical Systems, pages 1–7. IEEE, May 2025.*

#### Furthermore, if you utilize the datasets from this repository, please cite them as well:

CIC Dataset:

*Euclides Carlos Pinto Neto, Hamideh Taslimasa, Sajjad Dadkhah, Shahrear Iqbal, Pulei Xiong, Taufiq Rahman, and Ali A Ghorbani. Ciciov2024: Advancing realistic ids approaches against dos and spoofing attack in iov can bus. Internet of Things, 26:101209, 2024.*

HCRL Dataset:

*Hyun Min Song, Jiyoung Woo, and Huy Kang Kim. In-vehicle network intrusion detection using deep convolutional neural network. Vehicular Communications, 21:100198, 2020.*

## Getting Started

1. `pip install torch pandas numpy`
2. This repository comes with two pre-trained Transformer models for the HCRL and CIC dataset that are ready to be used in experiments.  They can be found in `final_models`
   - if you would like to add a dataset, it will need a config file with the same keys as the models in this repository
4. `python3 gumbel_softmax_tester.py` to run the space search with the following arguments:
   - `--model` (i.e. one of the models from `final_models/` with corresponding json config file)
   - `--model_path` (i.e. final_models)
   - `--dataset` cic or hcrl
   - `--output_dir` where you want the results saved to
   - `--exp_name` experiment name you want the output files to have (each file produced will be named `exp_name_<process_number>.txt`)
   - `--num_msgs` the number of messages that will be used for the space exploration (default 2500)
   - `--num_procs` the number of processes to be spun up to run the experiment in parallel (default 16)
