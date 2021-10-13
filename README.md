# UniSpeech

<!--**Pre-trained models for speech related tasks**-->

The family of UniSpeech:
> [**UniSpeech**](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech) (```ICML 2021```): **Unified Pre-training for Self-Supervised Learning and Supervised Learning for ASR**

> [**UniSpeech-SAT**](https://arxiv.org/pdf/2110.05752.pdf) (```ICASSP 2022 Submission```): **Universal Speech Representation Learning with  Speaker Aware Pre-Training**

## Update
- [Model Release] Octorber 13, 2021: [**UniSpeech-SAT**](https://arxiv.org/pdf/2110.05752.pdf) models are releaseed.
- [HuggingFace Integration] Octorber 11, 2021: [**UniSpeech**](https://huggingface.co/microsoft/layoutlm-base-casehttps://huggingface.co/microsoft/unispeech-large-1500h-cv)  models are on [HuggingFace](https://github.com/huggingface/transformers) . 
- [Model Release] June, 2021: [**UniSpeech v1**](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech) models are released.
## Pre-trained models
We strongly suggest using our UniSpeech-SAT model for speaker related tasks, since it shows very powerful performance on various speaker related benchmarks.
Model | Dataset | Model
|---|---|---
UniSpeech Base |  [1500 hrs CommonVoice](https://commonvoice.mozilla.org/) | [download](https://releasemodel.blob.core.windows.net/models/CommonVoicePretrainedModel/CommonVoiceEnglishPretrainedModel/checkpoint_best.pt?sv=2019-12-12&st=2021-07-14T09%3A00%3A07Z&se=2022-07-15T09%3A00%3A00Z&sr=b&sp=r&sig=5sxvEwVRoGtkazNQYkOuFLlPYau8nl5Ng%2FfRJa0Vnc4%3D)
UniSpeech Large |  [1500 hrs CommonVoice](https://commonvoice.mozilla.org/) | [download](https://releasemodel.blob.core.windows.net/models/CommonVoicePretrainedModel/CommonVoiceMultilingualPretrainedModel/checkpoint_best.pt?sv=2019-12-12&st=2021-07-14T09%3A00%3A39Z&se=2022-07-15T09%3A00%3A00Z&sr=b&sp=r&sig=y%2Fd3rqtbyqW0ZCwR7Czho5any90khA%2Ft3w9PTZ6N9vU%3D)
UniSpeech-SAT Base |  [960 hrs LibriSpeech](http://www.openslr.org/12) | [download](https://drive.google.com/file/d/1l5etRW6W2aP_8I2Fs_8ailGZqEzdrAPz/view?usp=sharing)
UniSpeech-SAT Base+ | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main) | [download](https://drive.google.com/file/d/1Q1MLVfyOHkSzTjyD-mzSZVjhndEmCvef/view?usp=sharing)
UniSpeech-SAT Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main) | [download](https://drive.google.com/file/d/12ScE1G2W-AHcccyBb_0uVI6qpFVQ0PaI/view?usp=sharing)

## Universal Representation Evaluation on SUPERB 
![alt text](UniSpeech-SAT/SUPERB_Results.png)

## Downstream Task Performance 
We also evaluate our models on typical speaker related benchmarks.
### Speaker Verification
| Model         |Fix pre-train| Vox1-O | Vox1-E     | Vox1-H         |
| ------------- |------------- | ---------- | ---------- | ---------- |
| ECAPA-TDNN   | - | 0.87     | 1.12  | 2.12   |
| HuBERT large  | Yes|  0.888	|0.912|	1.853 |
| Wav2Vec2.0 (XLSR)| Yes | 0.915|	0.945	|1.895|
| **UniSpeech-SAT large** | Yes | 0.771	| 0.781|	1.669|
| HuBERT large | No| 0.585|	0.654	|1.342|   
| Wav2Vec2.0 (XLSR) | No| 0.564|	0.605	|1.23|   
| **UniSpeech-SAT large** | No | **0.564** | **0.561** | **1.23** |

[paper for verification](https://arxiv.org/pdf/2110.05777.pdf)

Regarding reproduction, please contact [Zhengyang](https://github.com/czy97)


### Speech Separation

Evaluation on [LibriCSS](https://github.com/chenzhuo1011/libri_css)
| Model         |0S | 0L | OV10     |      OV20     |OV30 |OV40 |
| ---------------- |------| ------ | ------ | ------ | ------ | ------ |
| [Conformer](https://ieeexplore.ieee.org/abstract/document/9413423/) (SOTA)   | 4.5	| 4.4	|6.2	|8.5|	11	|12.6|
| **UniSpeech-SAT base** | 4.4|	4.4	|5.4|	7.2|	9.2	|10.5|
| **UniSpeech-SAT large** | 4.3|	4.2	|5.0	|6.3|	8.2|	8.8|

paper will appear soon

Regarding reproduction, please contact [Sanyuan](https://github.com/Sanyuan-Chen)

### Speech Diarization

Evaluation on CALLHOME
| Model         |spk_2	|spk_3|	spk_4|	spk_5|	spk_6|	spk_all |
| ---------------- |------| ------ | ------ | ------ | ------ | ------ |
| [EEND](https://arxiv.org/pdf/2105.09040.pdf) (SOTA)  | 7.96|	11.93	|16.38|	21.21|	23.1	|12.49||
| **UniSpeech-SAT large** | 5.93|	10.66|	12.9	|16.48|	23.25|	10.92|

paper will appear soon

Regarding reproduction, please contact [Zhengyang](https://github.com/czy97)

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)


### Reference
If you find Our work is useful in your research, please cite the following paper:

[UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597)

[UniSpeech-SAT: Universal Speech Representation Learning with  Speaker Aware Pre-Training](https://arxiv.org/pdf/2110.05752.pdf)

### Contact Information

For help or issues using UniSpeech models, please submit a GitHub issue.

For other communications related to UniSpeech, please contact Yu Wu (`yuwu1@microsoft.com`).