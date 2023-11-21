Note: the typical order in which you would run the scripts in this repository is the following: 
`preprocess_mimiv_iv.py` (once) $\to$ `culling.py` (once) $\to$ `training_local.py` $\to$ `evaluate_int_classif.py`.

# Prerequisites

You should have downloaded the MIMIV-IV dataset. It is available [here](https://physionet.org/content/mimiciv/2.2/). To be able to download it, you need to complet ea training, as indicated on the MIMIC website.

This dataset should then be placed in a `mimic-iv-2.2` folder, as such:

```
├── mimic-iv-2.2
│   ├── ...
├── STraTS-PyTorch 
│   ├── ...
```

# Use

A Docker environment is provided for simplicity. To build the image and run it, run

```
USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose up -d --build
```

A jupyter notebook can then be accessed inside the container (check ports in `docker-compose.yml` and adapt the jupyter password in `Dockerfile`).

