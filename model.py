import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    WhisperModel,
    get_linear_schedule_with_warmup,
    set_seed,
)
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import snapshot_download

class WhisperWithDiarization(
    nn.Module, 
    PyTorchModelHubMixin
):
    """
    Whisper model with CNN-based diarization head.
    """
    
    def __init__(self, whisper_model, num_diar_classes=3, hidden_dim=256, diar_loss_weight=1.0, 
                 encoder_output_mode='last_hidden', load_pretrained_diarization=True):
        """
        Args:
            whisper_model: WhisperForConditionalGeneration model
            num_diar_classes: Number of diarization classes (default: 3 for silence, child, adult)
            hidden_dim: Hidden dimension for CNN layers
            diar_loss_weight: Weight for diarization loss (default: 1.0). If 0.0, no diarization head will be created.
            encoder_output_mode: 'last_hidden' or 'weighted_sum' for combining encoder layers
            load_pretrained_diarization: Whether to load pretrained diarization weights from checkpoint (default: True, set to False when loading from fine-tuned model)
        """
        super().__init__()
        self.whisper = whisper_model
        self.config = whisper_model.config
        self.diar_loss_weight = diar_loss_weight
        self.encoder_output_mode = encoder_output_mode
        self.num_diar_classes = num_diar_classes
        self.has_diarization = diar_loss_weight > 0.0  # Flag to check if diarization is enabled
        
        # Get encoder hidden size
        encoder_hidden_size = self.config.d_model

        # Skip all diarization components if weight is 0.0
        if not self.has_diarization:
            self.layer_weights = None
            self.diarization_conv_layers = None
            self.diarization_post_conv = None
            self.diarization_classifier = None
            return
        
        
        
        # CNN-based diarization head
        # Input: [B, T, D] where D is encoder hidden size
        # Output: [B, T, num_classes]
        # Multi-layer CNN head
        # Split into parts for accessing embeddings before ReLU
        self.diarization_conv_layers = nn.Sequential(
            nn.Conv1d(encoder_hidden_size, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0),  # Last conv - we'll use output before ReLU
        )
        self.diarization_post_conv = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.diarization_classifier = nn.Conv1d(hidden_dim, num_diar_classes, kernel_size=1, padding=0)
        
 
    
    def _forward_with_conditioned_encoder(self, input_features, conditioned_encoder_hidden_states, labels, return_dict):
        """
        Forward pass through decoder with conditioned encoder hidden states.
        
        Args:
            input_features: Input mel spectrograms [B, n_mels, T]
            conditioned_encoder_hidden_states: Conditioned encoder output [B, T, D]
            labels: ASR labels
            return_dict: Whether to return dict
        
        Returns:
            Model outputs
        """
        # Create encoder outputs wrapper that the Whisper model expects
        from transformers.modeling_outputs import BaseModelOutput
        
        encoder_outputs = BaseModelOutput(
            last_hidden_state=conditioned_encoder_hidden_states,
            hidden_states=None,
            attentions=None
        )
        
        # Use Whisper's standard forward with our custom encoder outputs
        # Important: Don't pass decoder_input_ids, let Whisper derive it from labels
        outputs = self.whisper(
            input_features=None,
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=return_dict
        )
        
        return outputs
    
    def _build_conditioning_vector(self, diar_logits):
        """
        Build conditioning vector from diarization logits.
        
        Args:
            diar_logits: [B, T, num_classes] diarization logits
        
        Returns:
            c: [B, T, num_classes+1] conditioning vector with posteriors and confidence
        """
        # Compute posteriors from logits
        diar_posteriors = F.softmax(diar_logits, dim=-1)  # [B, T, K]
        
        # Compute entropy-based confidence: 1.0 - entropy / log(K)
        # Higher confidence when model is certain, lower when uncertain
        entropy = -torch.sum(diar_posteriors * torch.log(diar_posteriors + 1e-10), dim=-1)  # [B, T]
        max_entropy = torch.log(torch.tensor(self.num_diar_classes, dtype=entropy.dtype, device=entropy.device))
        confidence = 1.0 - entropy / max_entropy  # [B, T]
        confidence = confidence.unsqueeze(-1)  # [B, T, 1]
        
        # Concatenate: [posteriors, confidence]
        c = torch.cat([diar_posteriors, confidence], dim=-1)  # [B, T, K+1]
        
        return c
    
    def forward(self, input_features, labels=None, diar_labels=None,
                return_dict=True, output_hidden_states=True):
        """
        Forward pass with both ASR and diarization loss.
        
        Args:
            input_features: Input mel spectrograms [B, n_mels, T]
            labels: ASR labels [B, seq_len]
            diar_labels: Diarization labels [B, diar_seq_len] (padded with -100)
            return_dict: Whether to return dict
            output_hidden_states: Whether to output hidden states
        
        Returns:
            Dict with 'loss', 'asr_loss', 'diar_loss', 'logits', 'diar_logits'
        """
        # If diarization is disabled, just use standard Whisper forward
        if not self.has_diarization:
            outputs = self.whisper(
                input_features=input_features,
                labels=labels,
                return_dict=return_dict
            )
            return {
                'loss': outputs.loss,
                'asr_loss': outputs.loss if labels is not None else None,
                'diar_loss': None,
                'logits': outputs.logits,
                'diar_logits': None
            }
        
        # Get encoder outputs
        encoder_outputs = self.whisper.model.encoder(
            input_features,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # Get encoder hidden states based on mode
        if self.encoder_output_mode == 'weighted_sum' and encoder_outputs.hidden_states is not None:
            # Use weighted sum of all layers: [B, T, D]
            # hidden_states is a tuple of length (num_layers + 1) - includes embedding layer
            hidden_states_stack = torch.stack(encoder_outputs.hidden_states)  # [num_layers, B, T, D]
            # Apply softmax to layer weights for proper weighting
            layer_weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]
            encoder_hidden_states = torch.einsum('l,lbtd->btd', layer_weights, hidden_states_stack)
        else:
            # Use last hidden state from encoder: [B, T, D]
            encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Compute diarization embeddings and logits
        # Transpose for Conv1d: [B, T, D] -> [B, D, T]
        diar_input = encoder_hidden_states.transpose(1, 2)
        
        # Complex mode: multi-layer with intermediate embeddings
        diar_embeddings_pre_relu = self.diarization_conv_layers(diar_input)  # [B, hidden_dim, T] - before ReLU
        diar_embeddings_post_relu = self.diarization_post_conv(diar_embeddings_pre_relu)  # [B, hidden_dim, T] - after ReLU
        diar_logits = self.diarization_classifier(diar_embeddings_post_relu)  # [B, num_classes, T]
        diar_logits = diar_logits.transpose(1, 2)  # [B, T, num_classes]
        
        # Compute ASR output using standard forward
        outputs = self.whisper(
            input_features=input_features,
            labels=labels,
            return_dict=return_dict
        )
        
        asr_loss = outputs.loss if labels is not None else None
        
        # Compute diarization loss if labels provided
        diar_loss = None
        if diar_labels is not None:
            # Flatten for cross entropy
            # diar_logits: [B, T, C], diar_labels: [B, T]
            batch_size, seq_len, num_classes = diar_logits.shape
            
            # Create mask for valid positions (not -100)
            valid_mask = (diar_labels != -100)
            if valid_mask.any():
                # Compute loss only on valid positions
                diar_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                diar_loss = diar_loss_fn(
                    diar_logits.reshape(-1, num_classes),
                    diar_labels.reshape(-1)
                )
        
        # Combine losses
        total_loss = None
        if asr_loss is not None and diar_loss is not None:
            # Weight the diarization loss
            total_loss = asr_loss + self.diar_loss_weight * diar_loss
        elif asr_loss is not None:
            total_loss = asr_loss
        elif diar_loss is not None:
            total_loss = self.diar_loss_weight * diar_loss
        
        return {
            'loss': total_loss,
            'asr_loss': asr_loss,
            'diar_loss': diar_loss,
            'logits': outputs.logits,
            'diar_logits': diar_logits
        }
    
    def generate(self, input_features, **kwargs):
        """
        Generate transcriptions using the model.
        """
        # Use standard Whisper generation
        return self.whisper.generate(input_features, **kwargs)
    
    def save_pretrained(self, save_directory, config=None, **kwargs):
        """Save both Whisper model and diarization head"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save Whisper model
        self.whisper.save_pretrained(save_directory)
        
        # Skip diarization components if not using diarization
        if not self.has_diarization:
            torch.save(False, os.path.join(save_directory, 'has_diarization.pt'))
            return
        
        # Save diarization components
        diar_conv_path = os.path.join(save_directory, 'diarization_conv_layers.pt')
        torch.save(self.diarization_conv_layers.state_dict(), diar_conv_path)
        
        # Save classifier only if not simple diarization
        diar_classifier_path = os.path.join(save_directory, 'diarization_classifier.pt')
        torch.save(self.diarization_classifier.state_dict(), diar_classifier_path)
        
        torch.save(True, os.path.join(save_directory, 'has_diarization.pt'))
    
    @classmethod
    def from_pretrained(cls, repo_id, whisper_model_class=WhisperForConditionalGeneration,
                        num_diar_classes=3, hidden_dim=256, diar_loss_weight=1.0, 
                        encoder_output_mode='last_hidden', gate_hidden_dim=256, config=None):
        """Load both Whisper model and diarization head from fine-tuned checkpoint"""
        # Load Whisper model
        whisper_model = whisper_model_class.from_pretrained(repo_id)
        
        local_dir = snapshot_download(repo_id=repo_id)
        # Create model WITHOUT loading baseline diarization weights
        # We'll load the fine-tuned weights from load_directory instead
        model = cls(
            whisper_model,
            num_diar_classes=num_diar_classes,
            hidden_dim=hidden_dim,
            diar_loss_weight=diar_loss_weight,
            encoder_output_mode=encoder_output_mode,
            load_pretrained_diarization=False,  # Don't load baseline weights!
        )
        
        # Load diarization components
        diar_conv_path = os.path.join(local_dir, 'diarization_conv_layers.pt')
        diar_classifier_path = os.path.join(local_dir, 'diarization_classifier.pt')
        
        model.diarization_conv_layers.load_state_dict(torch.load(diar_conv_path))
        
        model.diarization_classifier.load_state_dict(torch.load(diar_classifier_path))
        
        return model