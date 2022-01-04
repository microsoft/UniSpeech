
# ILS-SSL

> [**WavLM**](https://arxiv.org/pdf/2112.08778.pdf): Self-Supervised Learning for Speech Recognition with Intermediate Layer Supervision

The data preparation and pre-training for the first iteration follow the same pipeline as Hubert. We give example scripts for ILS-Hubert pre-training and fine-tuning in src/examples/hubert/scripts

## Pre-Trained and Fine-tuned Models
Model | Pretraining Dataset | Finetuning Dataset | Model
|---|---|---|---
ILS-Base | 960h LibriSpeech | - |comming soon
ILS-Large | 60k hrs Libri-Light | - |comming soon


## Results on Librispeech
Model | Finetuning set|  LM | test-clean | test-other
|---|---|---|---|---
wav2vec2.0 Base | 1 hour | None |  24.5 | 29.7
Hubert Base | 1 hour | None| 20.9 | 27.5
ILS-SSL Base | 1 hour | None | 17.9 | 23.1
wav2vec2.0 Base | 1 hour | 4-gram | 5.5 | 11.3
Hubert Base | 1 hour | 4-gram | 6.1 | 11.3
ILS-SSL Base | 1 hour | 4-gram | 5.4 | 10.2
wav2vec2.0 Base | 10 hour | None | 11.1 | 17.6
Hubert Base | 10 hour | None| 10.1 | 16.8
ILS-SSL Base | 10 hour | None | 8.3 | 13.6
wav2vec2.0 Base | 10 hour | 4-gram | 4.3 | 9.5
Hubert Base | 10 hour | 4-gram | 4.3 | 9.4
ILS-SSL Base | 10 hour | 4-gram | 3.8 | 8.1
wav2vec2.0 Base | 100 hour | None | 6.1 | 13.3
Hubert Base | 100 hour | None| 6.3 | 13.2
ILS-SSL Base | 100 hour | None | 4.7 | 10.1
wav2vec2.0 Base | 100 hour | 4-gram | 3.4| 8.0
Hubert Base | 100 hour | 4-gram | 3.4 | 8.1
ILS-SSL Base | 100 hour | 4-gram | 3.0 | 6.9
