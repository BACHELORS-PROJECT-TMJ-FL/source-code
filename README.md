# Bachelors project

This repository contains source code for bachelors project:
"A Federated Learning Approach to Early Onset
Detection of Temporomandibular Joint Involvement
in Juvenile Idiopathic Arthritis Patients".

The project has been done and completed by Mathias Jørgensen and Daniel Østerballe

## Setup

This project uses Python 3.11.9.
To install the necessary python packages, we recommend setup virtual environment (commands for windows):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Overview

- TMJ-ML: For running centralized learning (flower) on the TMJ dataset
- TMJ-FL: Running federated learning on the TMJ dataset
  - A vital module is missing from flwrapp, that is the data processing pipeline developed by Lena and Dicte. This code is not public, and therefore not be included here.
- MNIST-FL: For running experiment 3 on the MNIST dataset utilizaing Flower framework
