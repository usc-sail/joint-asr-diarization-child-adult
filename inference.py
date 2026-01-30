#!/usr/bin/env python3
"""
Batch inference script for fine-tuned Whisper models on all Playlogue test sets
Processes all test audio files and saves speaker and timestamp tagged transcripts
"""

import torch
import librosa
import argparse
from transformers import WhisperProcessor, LogitsProcessorList
from structured_logits_processor import StructuredOutputLogitsProcessor
from silence_masking_processor import SilenceMaskingProcessor
from model import WhisperWithDiarization


def get_vad_outputs(model, input_features, device='cpu', silence_threshold=0.7):
    """
    Get VAD (Voice Activity Detection) outputs from the model to detect silence segments.
    
    Args:
        model: WhisperWithDiarization model
        input_features: Input mel spectrograms [B, n_mels, T]
        device: Device to run on
        silence_threshold: Probability threshold for determining silence (default: 0.7)
    
    Returns:
        silence_segments: List of silence segments with start/end times
    """
    if not isinstance(model, WhisperWithDiarization):
        return []
    
    model.eval()
    with torch.no_grad():
        # Get encoder outputs
        encoder_outputs = model.whisper.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Always use last hidden state from encoder: [B, T, D]
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Compute diarization logits
        diar_input = encoder_hidden_states.transpose(1, 2)  # [B, D, T]
        
        # Check if model has new structure (diarization_conv_layers) or old structure (diarization_head)
        if hasattr(model, 'diarization_conv_layers'):
            diar_embeddings_pre_relu = model.diarization_conv_layers(diar_input)  # [B, hidden_dim, T] or [B, num_classes, T]
            
            diar_embeddings_post_relu = model.diarization_post_conv(diar_embeddings_pre_relu)  # [B, hidden_dim, T]
            diar_logits = model.diarization_classifier(diar_embeddings_post_relu)  # [B, num_classes, T]
        else:
            diar_logits = model.diarization_head(diar_input)  # [B, num_classes, T]
        
        diar_logits = diar_logits.transpose(1, 2)  # [B, T, num_classes]
        
        # Get probabilities
        diar_probs = torch.softmax(diar_logits, dim=-1)  # [B, T, num_classes]
        
        # Extract silence probabilities (assuming class 0 is silence)
        silence_probs = diar_probs[0, :, 0].cpu().numpy()  # [T]
    
    # Convert frame-level silence probabilities to segments
    # Each frame is ~20ms (Whisper encoder downsamples by 2)
    frame_duration_s = 0.02
    
    silence_segments = []
    in_silence = False
    silence_start = None
    
    for frame_idx, prob in enumerate(silence_probs):
        if prob >= silence_threshold:
            # This frame is silence
            if not in_silence:
                # Start of new silence segment
                in_silence = True
                silence_start = frame_idx * frame_duration_s
        else:
            # This frame is not silence
            if in_silence:
                # End of silence segment
                silence_end = frame_idx * frame_duration_s
                silence_segments.append({
                    'start': round(silence_start, 2),
                    'end': round(silence_end, 2)
                })
                in_silence = False
                silence_start = None
    
    # Handle case where silence extends to the end
    if in_silence and silence_start is not None:
        silence_end = len(silence_probs) * frame_duration_s
        silence_segments.append({
            'start': round(silence_start, 2),
            'end': round(silence_end, 2)
        })
    
    return silence_segments


def transcribe_audio(model, processor, audio_path, device='cuda', enable_silence_masking=True, enable_logits_processors=True, max_len=300):
    """Transcribe audio and return speaker/timestamp tagged output"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Process audio
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    silence_segments = []
    
    if enable_silence_masking:
        # Use VAD to get silence segments with 0.7 threshold
        silence_segments = get_vad_outputs(model, input_features, device, silence_threshold=0.7)
        
        # Filter silence segments to keep only those ≥1.0s
        silence_segments = [seg for seg in silence_segments if (seg['end'] - seg['start']) >= 1.0]
    
    # Get token IDs
    start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    notimestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    prompt = "<|startoftranscript|><|en|><|transcribe|>"
    # Use decoder_input_ids to force the correct sequence like in training
    # Start with the correct sequence: [start_token, en_token, transcribe_token]
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    # Build logits processor list
    logits_processors = []
    
    if enable_logits_processors:
        logits_processors.append(
            StructuredOutputLogitsProcessor(
                tokenizer=processor.tokenizer,
                speaker_tokens=("<adult>", "<child>"),   # <-- your exact tokens
                max_len=max_len,
                device=str(device)
            )
        )
        # import pdb; pdb.set_trace()
        # Add silence masking processor if we have silence segments
        if len(silence_segments) > 0:
            silence_masking = SilenceMaskingProcessor(
                tokenizer=processor.tokenizer,
                silence_segments=silence_segments,
                buffer_s=0.2  # 0.2s buffer on each side
            )
            logits_processors.append(silence_masking)
    
    logits_proc = LogitsProcessorList(logits_processors)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            input_features, 
            logits_processor=logits_proc,
            decoder_input_ids=decoder_input_ids,
            max_length=max_len,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            suppress_tokens=[notimestamps_token_id, start_token_id] if notimestamps_token_id is not None else [],
            early_stopping=False,
            repetition_penalty=1.1,
            return_timestamps=False
        )
        token_ids = generated_tokens[0].cpu().numpy()
        if len(token_ids) == max_len:
            token_ids = token_ids[:3]
            transcript = decode_with_timestamps(processor, token_ids) + "reach_max_length"
        else:
            transcript = decode_with_timestamps(processor, token_ids)
        
    return transcript


def decode_with_timestamps(processor, token_ids):
    """Decode tokens while preserving speaker tags and timestamps"""
    result_parts = []
    
    for token_id in token_ids:
        token_str = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
        
        # Check if it's a timestamp token
        if token_str.startswith('<|') and token_str.endswith('|>'):
            result_parts.append(token_str)
        # Skip control tokens
        elif token_str in ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|endoftranscript|>']:
            continue
        else:
            # Regular text token
            decoded_text = processor.tokenizer.decode([token_id])
            if decoded_text.strip():
                result_parts.append(decoded_text)
    
    return ''.join(result_parts)


def main():
    parser = argparse.ArgumentParser(description='Batch Whisper inference for all Playlogue test sets')
    parser.add_argument('--wav-file', type=str,
                        help='Path to wav file to transcribe')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    parser.add_argument('--disable-silence-masking', dest='enable_silence_masking', action='store_false', default=True,
                        help='Disable silence masking during inference (silence masking is enabled by default for diarization models)')
    parser.add_argument('--disable-logits-processors', dest='enable_logits_processors', action='store_false', default=True,
                        help='Disable all logits processors during inference (logits processors are enabled by default)')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    print(f"Using device: {device}")
    
    # Load model
    processor = WhisperProcessor.from_pretrained('AlexXu811/child-adult-joint-asr-diarization')
    model = WhisperWithDiarization.from_pretrained('AlexXu811/child-adult-joint-asr-diarization')
    print("=" * 60)
    print("DIARIZATION MODEL DETECTED")
    print(f"Silence masking: {'ENABLED' if args.enable_silence_masking else 'DISABLED'}")
    print(f"Logits processors: {'ENABLED' if args.enable_logits_processors else 'DISABLED'}")
    print("Will output both ASR transcripts and speaker diarization")
    print("=" * 60)
    
    
    prediction = transcribe_audio(
        model,
        processor,
        args.wav_file,
        device,
        args.enable_silence_masking,
        args.enable_logits_processors
    )
    prediction = prediction.replace("<|startoftranscript|><|en|><|transcribe|>", "").strip()
    print(f"  Prediction: {prediction}")


if __name__ == "__main__":
    main()