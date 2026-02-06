# Joint ASR/Diarization for child-adult interactions
Inference code for the paper "End-to-End Joint ASR and Speaker Role Diarization with Child-Adult Interactions."

Preprint: https://arxiv.org/abs/2601.17640.


## Public Model Weights
The model weights are released at https://huggingface.co/AlexXu811/child-adult-joint-asr-diarization.
The released weights are trained on the public Playlogue dataset using Whisper-small.en. 

## Set up the environment
```
conda create -n joint_asr_diar_child python==3.10
conda activate joint_asr_diar_child
pip install -r requirements.txt 
```

## Example Inference Code
Currently, it only supports WAV files up to 30 seconds due to Whisper's maximum audio input length limit. Please chunk your audio into chunks with <30s lengths (we might release the code to integrate this step in the future).

```
import torch
from transformers import WhisperProcessor
from model import WhisperWithDiarization
from inference import transcribe_audio

wav_file = "synthetic_example.wav"  # Path to your audio file

processor = WhisperProcessor.from_pretrained('AlexXu811/child-adult-joint-asr-diarization')
model = WhisperWithDiarization.from_pretrained('AlexXu811/child-adult-joint-asr-diarization')

prediction = transcribe_audio(
    model,
    processor,
    wav_file
)

print(prediction.replace("<|startoftranscript|><|en|><|transcribe|>", "").strip())
```

## Citation
```
@article{xu2026end,
  title={End-to-End Joint ASR and Speaker Role Diarization with Child-Adult Interactions},
  author={Xu, Anfeng and Feng, Tiantian and Bishop, Somer and Lord, Catherine and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2601.17640},
  year={2026}
}
```
