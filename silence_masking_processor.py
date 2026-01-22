#!/usr/bin/env python3
"""
Silence masking logits processor that suppresses timestamp tokens
corresponding to silence regions identified by diarization.
"""

import torch
import re
from transformers import LogitsProcessor


class SilenceMaskingProcessor(LogitsProcessor):
    """
    Masks timestamp tokens that fall within silence regions (with buffer).
    
    This processor uses diarization predictions to identify silence regions
    and prevents the model from generating timestamps within those regions.
    Only applies masking during T1 (timestamp) prediction state.
    """
    
    def __init__(self, tokenizer, silence_segments, buffer_s=0.2, speaker_tokens=("<adult>", "<child>")):
        """
        Args:
            tokenizer: Whisper tokenizer
            silence_segments: List of dicts with 'start' and 'end' times (in seconds)
                              representing silence regions to mask
            buffer_s: Buffer time in seconds to subtract from beginning and end
                      (e.g., for a 2s silence segment, with 0.5s buffer, 1s in the middle is masked)
            speaker_tokens: Tuple of speaker token strings for state detection
        """
        self.tokenizer = tokenizer
        self.buffer_s = buffer_s
        
        # Get vocabulary for state detection
        vocab = tokenizer.get_vocab()
        
        # Find timestamp tokens
        ts_pattern = re.compile(r"<\|(\d+\.\d+)\|>")
        self.timestamp_ids = set()
        
        # Find speaker tokens for state detection
        self.speaker_ids = set()
        for token in speaker_tokens:
            if token in vocab:
                self.speaker_ids.add(vocab[token])
        
        # Process silence segments - apply buffer
        self.masked_regions = []
        for seg in silence_segments:
            duration = seg['end'] - seg['start']
            
            # Determine buffer application
            # Only add start buffer if silence doesn't start at beginning (0.0)
            start_buffer = 0.0 if seg['start'] == 0.0 else buffer_s
            end_buffer = buffer_s
            
            total_buffer = start_buffer + end_buffer
            
            if duration > total_buffer:  # Only mask if there's something left after buffers
                # Add buffer: move start forward (if not at beginning) and end backward
                masked_start = seg['start'] + start_buffer
                masked_end = seg['end'] - end_buffer
                if masked_start < masked_end:
                    self.masked_regions.append((masked_start, masked_end))
        
        # Get all timestamp tokens and their corresponding times
        self.timestamp_to_id = {}
        self.masked_timestamp_ids = set()
        
        for token, tid in vocab.items():
            match = ts_pattern.fullmatch(token)
            if match:
                time_s = float(match.group(1))
                self.timestamp_to_id[time_s] = tid
                self.timestamp_ids.add(tid)
                
                # Check if this timestamp falls within any masked region
                for masked_start, masked_end in self.masked_regions:
                    if masked_start <= time_s <= masked_end:
                        self.masked_timestamp_ids.add(tid)
                        break
        
        print(f"Silence masking: {len(self.masked_regions)} regions, {len(self.masked_timestamp_ids)} timestamps masked")
    
    def _get_state(self, seq_ids):
        """State determination for T1 -> SPEAKER -> TEXT -> T2 -> T1 or end"""
        if len(seq_ids) <= 3:
            return "need_t1"
        recent = seq_ids[-6:].cpu().tolist() if len(seq_ids) >= 6 else seq_ids.cpu().tolist()
        if len(recent) == 0:
            return "need_t1"
        last_token = recent[-1]
        
        if last_token in self.timestamp_ids:
            # Check what came before to determine if this is T1 or T2
            if len(recent) >= 2 and recent[-2] in self.speaker_ids:
                # This should not happen in our new flow
                return "need_text"
            elif len(recent) >= 2 and recent[-2] not in self.speaker_ids and recent[-2] not in self.timestamp_ids:
                # After TEXT, this is T2
                return "need_t1"  # After T2, need T1 or end
            else:
                # This is T1 at the start or after T2
                return "need_speaker"
        elif last_token in self.speaker_ids:
            # After SPEAKER, need TEXT
            return "need_text"
        elif last_token not in self.speaker_ids and last_token not in self.timestamp_ids:
            # Last token is TEXT
            if len(recent) >= 2 and recent[-2] in self.speaker_ids:
                # First token after SPEAKER
                decoded = self.tokenizer.decode([last_token])
                if decoded.strip() == '':
                    return "need_text"  # Still need non-space text
            return "in_or_t2"  # Can continue text or go to T2
        else:
            return "need_t1"  # Fallback
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Mask timestamp tokens that fall within silence regions.
        Only applies masking during T1 (timestamp) prediction state.
        """
        if len(self.masked_timestamp_ids) == 0:
            return scores
        
        # Apply masking only when in "need_t1" state
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            seq = input_ids[b]
            state = self._get_state(seq)
            
            # Only mask silence timestamps when predicting T1 (timestamp)
            # if state == "need_t1":
            for tid in self.masked_timestamp_ids:
                scores[b, tid] = float("-inf")
        
        return scores
