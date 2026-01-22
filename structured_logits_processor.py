# structured_logits_processor.py
import re
import torch
from transformers import LogitsProcessor

class StructuredOutputLogitsProcessor(LogitsProcessor):
    """
    Enforces the token cycle:
    T1 (timestamp) → SPEAKER → need TEXT (first non-space token) → in TEXT (continue text or go to T2) → T2 (timestamp) → (T1 or end)
    - "need TEXT": expects the first non-space text token after SPEAKER
    - "in TEXT": allows continuation of text or transition to T2
    Simple and robust implementation.
    """
    def __init__(
        self,
        tokenizer,
        max_len = 300,
        speaker_tokens=("<adult>", "<child>"),
        device="cuda",
    ):
        self.tok = tokenizer
        self.max_len = max_len
        vocab = tokenizer.get_vocab()
        
        # Find timestamp tokens - only valid ones
        ts_pat = re.compile(r"<\|\d+\.\d+\|>")
        self.timestamp_ids = set()
        for token, tid in vocab.items():
            if ts_pat.fullmatch(token):
                self.timestamp_ids.add(tid)
        
        # Find speaker tokens
        self.speaker_ids = set()
        for token in speaker_tokens:
            if token in vocab:
                self.speaker_ids.add(vocab[token])

    def _get_state(self, seq_ids):
        """State determination for T1 -> SPEAKER -> TEXT -> T2 -> T1 or end"""
        if len(seq_ids) <= 3:
            return "need_t1"
        recent = seq_ids[-6:].cpu().tolist() if len(seq_ids) >= 6 else seq_ids.cpu().tolist()
        if len(recent) == 0:
            return "need_t1"
        last_token = recent[-1]
        # State transitions:
        # 1. Start: need T1 (timestamp)
        # 2. After T1: need SPEAKER
        # 3. After SPEAKER: need TEXT (first non-space token)
        # 4. After TEXT: in TEXT (continue text or go to T2)
        # 5. After T2: need T1 (timestamp) or end
        #
        # Find last speaker, last timestamp
        if last_token in self.timestamp_ids:
            # Check what came before to determine if this is T1 or T2
            if len(recent) >= 5 and recent[-2] in self.speaker_ids:
                # This should not happen in our new flow
                # After SPEAKER we need TEXT, not timestamp
                return "need_text"
            elif len(recent) >= 5 and recent[-2] not in self.speaker_ids and recent[-2] not in self.timestamp_ids:
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
            # Check if this is the first text token after SPEAKER
            if len(recent) >= 5 and recent[-2] in self.speaker_ids:
                # First token after SPEAKER
                decoded = self.tok.decode([last_token])
                if decoded.strip() == '':
                    return "need_text"  # Still need non-space text
            return "in_or_t2"  # Can continue text or go to T2
        else:
            return "need_t1"  # Fallback

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """Structured masking for T1 -> SPEAKER -> TEXT -> T2 -> T1 or end"""
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            seq = input_ids[b]
            state = self._get_state(seq)
            if len(seq) >= self.max_len - 1:
                # force timestamp
                mask = torch.zeros(scores.size(1), dtype=torch.bool, device=scores.device)
                for tid in self.timestamp_ids:
                    mask[tid] = True
                mask[self.tok.eos_token_id] = True
                scores[b] = scores[b].masked_fill(~mask, float("-inf"))
            # Apply masking based on state
            if state == "need_t1":
                # Only allow timestamps or end
                mask = torch.zeros(scores.size(1), dtype=torch.bool, device=scores.device)
                for tid in self.timestamp_ids:
                    mask[tid] = True
                mask[self.tok.eos_token_id] = True
                scores[b] = scores[b].masked_fill(~mask, float("-inf"))
            elif state == "need_speaker":
                # Only allow speakers
                mask = torch.zeros(scores.size(1), dtype=torch.bool, device=scores.device)
                for sid in self.speaker_ids:
                    mask[sid] = True
                scores[b] = scores[b].masked_fill(~mask, float("-inf"))
            elif state == "need_text":
                # Only allow text (not speaker, not timestamp)
                for tid in self.timestamp_ids:
                    scores[b, tid] = float("-inf")
                for sid in self.speaker_ids:
                    scores[b, sid] = float("-inf")
                scores[b, self.tok.eos_token_id] = float("-inf")
            elif state == "in_or_t2":
                # Allow text (not speaker, not timestamp) or T2 (timestamp)
                mask = torch.ones(scores.size(1), dtype=torch.bool, device=scores.device)
                # Forbid speakers
                for sid in self.speaker_ids:
                    mask[sid] = False
                mask[self.tok.eos_token_id] = False
                # Allow timestamps (T2)
                # (all timestamps allowed, but only T2 will be chosen by model)
                # So do not mask out timestamps
                scores[b] = scores[b].masked_fill(~mask, float("-inf"))
        return scores
