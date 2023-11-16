## Pre-training Representations for Speaker Verification

### Pre-trained models

| Model                                                        | Fix pre-train | Vox1-O    | Vox1-E    | Vox1-H   |
| ------------------------------------------------------------ | ------------- | --------- | --------- | -------- |
| [ECAPA-TDNN](https://drive.google.com/file/d/1kWmLyTGkBExTdxtwmrXoP4DhWz_7ZAv3/view?usp=sharing) | -             | 1.080     | 1.200     | 2.127    |
| [HuBERT large](https://valle.blob.core.windows.net/share/unispeech-sat/sv_fix/HuBERT_large_SV_fixed.th?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) | Yes           | 0.888     | 0.912     | 1.853    |
| [Wav2Vec2.0 (XLSR)](https://valle.blob.core.windows.net/share/unispeech-sat/sv_fix/wav2vec2_xlsr_SV_fixed.th?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) | Yes           | 0.915     | 0.945     | 1.895    |
| [UniSpeech-SAT large](https://valle.blob.core.windows.net/share/unispeech-sat/sv_fix/UniSpeech-SAT_large_SV_fixed.th?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) | Yes           | 0.771     | 0.781     | 1.669    |
| [WavLM Base](https://valle.blob.core.windows.net/share/wavlm/sv_fix/wavlm_base_plus_nofinetune.pth?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) | Yes             | 0.84     | 0.928     | 1.758    |
| [**WavLM large**](https://valle.blob.core.windows.net/share/wavlm/sv_fix/wavlm_large_nofinetune.pth?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) | Yes           | 0.75     | 0.764     | 1.548    |
| [HuBERT large](https://drive.google.com/file/d/1nit9Z6RyM8Sdb3n8ccaglOQVNnqsjnui/view?usp=sharing) | No            | 0.585     | 0.654     | 1.342    |
| [Wav2Vec2.0 (XLSR)](https://drive.google.com/file/d/1TgKro9pp197TCgIF__IlE_rMVQOk50Eb/view?usp=sharing) | No            | 0.564     | 0.605     | 1.23     |
| [UniSpeech-SAT large](https://drive.google.com/file/d/10o6NHZsPXJn2k8n57e8Z_FkKh3V4TC3g/view?usp=sharing) | No            | 0.564 | 0.561 | 1.23 |
| [**WavLM large**](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing) | No            | **0.431** | **0.538** | **1.154** |

### How to use?

#### Environment Setup

1. `pip install --require-hashes -r requirements.txt`
2. Install fairseq code
   - For HuBERT_Large and Wav2Vec2.0 (XLSR), we should install the official [fairseq](https://github.com/pytorch/fairseq).
   - For UniSpeech-SAT large, we should install the [Unispeech-SAT](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT) fairseq code.
   - For WavLM, we should install the latest s3prl: `pip install s3prl@git+https://github.com/s3prl/s3prl.git@7ab62aaf2606d83da6c71ee74e7d16e0979edbc3#egg=s3prl`

#### Example

Take `unispeech_sat ` and `ecapa_tdnn` for example:

1. First, you should download the pre-trained model in the above table to `checkpoint_path`.
2. Then, run the following codes:
   - The wav files are sampled from [voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).

```bash
python verification.py --model_name unispeech_sat --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/Josh_Gad/HXUqYaOwrxA_0000015.wav --checkpoint $checkpoint_path
# output: The similarity score between two audios is 0.0317 (-1.0, 1.0).

python verification.py --model_name unispeech_sat --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/David_Faustino/xTOk1Jz-F_g_0000015.wav --checkpoint --checkpoint $checkpoint_path
# output: The similarity score between two audios is 0.5389 (-1.0, 1.0).

python verification.py --model_name ecapa_tdnn --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/Josh_Gad/HXUqYaOwrxA_0000015.wav --checkpoint $checkpoint_path
# output: The similarity score between two audios is 0.2053 (-1.0, 1.0).

python verification.py --model_name ecapa_tdnn --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/David_Faustino/xTOk1Jz-F_g_0000015.wav --checkpoint --checkpoint $checkpoint_path
# output: he similarity score between two audios is 0.5302 (-1.0, 1.0).
```

