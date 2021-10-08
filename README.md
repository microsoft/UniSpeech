# UniSpeech

<!--**Pre-trained models for speech related tasks**-->

The family of UniSpeech:
> [**UniSpeech**](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech) (```ICML 2021```): **Unified Pre-training for Self-Supervised Learning and Supervised Learning for ASR**

> [**UniSpeech-SAT**](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT) (```ICASSP 2022 Submission```): **Universal Speech Representation Learning with  Speaker Aware Pre-Training**

## Pre-trained models
We strongly suggest using our UniSpeech-SAT model for speaker related tasks, since it shows very powerful performance on various speaker related benchmarks.
Model | Dataset | Model
|---|---|---
UniSpeech Base |  [1500 hrs CommonVoice](https://commonvoice.mozilla.org/) | [download](https://releasemodel.blob.core.windows.net/models/CommonVoicePretrainedModel/CommonVoiceEnglishPretrainedModel/checkpoint_best.pt?sv=2019-12-12&st=2021-07-14T09%3A00%3A07Z&se=2022-07-15T09%3A00%3A00Z&sr=b&sp=r&sig=5sxvEwVRoGtkazNQYkOuFLlPYau8nl5Ng%2FfRJa0Vnc4%3D)
UniSpeech Large |  [1500 hrs CommonVoice](https://commonvoice.mozilla.org/) | [download](https://releasemodel.blob.core.windows.net/models/CommonVoicePretrainedModel/CommonVoiceMultilingualPretrainedModel/checkpoint_best.pt?sv=2019-12-12&st=2021-07-14T09%3A00%3A39Z&se=2022-07-15T09%3A00%3A00Z&sr=b&sp=r&sig=y%2Fd3rqtbyqW0ZCwR7Czho5any90khA%2Ft3w9PTZ6N9vU%3D)
UniSpeech-SAT Base |  [960 hrs LibriSpeech](http://www.openslr.org/12) | [download](https://drive.google.com/file/d/1l5etRW6W2aP_8I2Fs_8ailGZqEzdrAPz/view?usp=sharing)
UniSpeech-SAT Base+ | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main) | [download](https://drive.google.com/file/d/1Q1MLVfyOHkSzTjyD-mzSZVjhndEmCvef/view?usp=sharing)
UniSpeech-SAT Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main) | [download](https://drive.google.com/file/d/12ScE1G2W-AHcccyBb_0uVI6qpFVQ0PaI/view?usp=sharing)


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using UniSpeech models, please submit a GitHub issue.

For other communications related to UniSpeech, please contact Yu Wu (`yuwu1@microsoft.com`).