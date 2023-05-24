Introduction
---
This repository contains implementation of the [SimCLR][1] in PyTorch for my thesis in university.


Setup Dependencies
---
For running the experiments I have used Python 3.7, libs and versions are in ``requirements.txt`` file.

Also if you want to train with cuda enabled, you should install torch with cuda. Here is example if you have cuda 11.6 installed:
```bash 
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Configuration
---
You can pass command line arguments to the files ``train.py`` for simclr training and ``eval.py``  for linear classifier evaluation on top of the learned representations.

Arguments can be found in ``parser_settings.py``

Example of usage:

This command would do the self-supervised training on the dataset at `datasets/oral_dataset` and store the results in the `results` directory. The training batch size is 128, also cuda is used.
```bash
python train.py -datapath "datasets/oral_dataset" -respath results -ep 10 -bs 128 -c
```

This command would run the linear evaluator, for which the dataset is at `datasets/oral_dataset`, stored self-supervised model is at `results/model/model.pth` and would produce the results at `results/plots` (if any), uses cuda. Batch size used is 128.

```bash
python eval.py -datapath "datasets/oral_dataset" -model_path "results/model/model.pth" -respath results -bs 128 -c
```

Also, you need to configure working directory to run this files from `diploma_implementation` to get correct results.

## Datasets
I have used the two datasets: [Laryngeal dataset][2] and [Pharyngitis dataset][3], which edited for use and stored at `datasets` folder. For linear evaluation, I have used 10% of the train set images.


## Results

### Laryngeal dataset trained for 500 epochs

| class    | precision | recall | f1-score | support |
|----------|-----------|--------|----------|---------|
| Hbv      | 0.9242    | 0.9531 | 0.9384   | 64      |
| He       | 0.8039    | 0.9411 | 0.8817   | 42      |
| IPCL     | 0.9697    | 0.7200 | 0.9552   | 68      |
| Le       | 0.9445    | 0.7556 | 0.8395   | 45      |
| Accuracy |           |        |          | 0.9132  |

### Pharyngitis dataset trained for 500 epochs

| class          | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| pharyngitis    | 0.6667    | 0.5455 | 0.6      | 64      |
| no_pharyngitis | 0.7298    | 0.8182 | 0.7714   | 42      |
| Accuracy       |           |        |          | 0.7091  |


You can find pretrained models with optimizer and loss file for these datasets [here][4] (number of epochs: 10, 50, 100, 500).

[1]:https://arxiv.org/pdf/2002.05709.pdf 
[2]:https://zenodo.org/record/1003200
[3]:https://data.mendeley.com/datasets/8ynyhnj2kz
[4]:https://drive.google.com/drive/folders/1LHe5cy2LK5czuiycleOMzB5dmJKQiHvn?usp=sharing



