"""
Explicit Reasoning Module for EVA.
Adds chain-of-thought reasoning with thinking tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReasoningTokens:
    """Special tokens for chain-of-thought reasoning."""
    THINK = 65536  # <think>
    STEP = 65537   # <step>
    ANSWER = 65538 # <answer>
    END = 65539    # </think>


class ReasoningMemory(nn.Module):
    """
    Explicit reasoning memory for chain-of-thought.
    Stores intermediate reasoning steps in a dedicated buffer.
    """
    def __init__(self, D, max_steps=8):
        super().__init__()
        self.D = D
        self.max_steps = max_steps

        # Reasoning step encoder
        self.step_encoder = nn.Linear(D, D)

        # Reasoning attention (attend to previous steps)
        self.step_query = nn.Linear(D, D)
        self.step_key = nn.Linear(D, D)
        self.step_value = nn.Linear(D, D)

        # Output projection
        self.output_proj = nn.Linear(D, D)

    def forward(self, h, reasoning_buffer=None):
        """
        h: (B, L, D) current hidden state
        reasoning_buffer: list of previous reasoning step tensors
        """
        B, L, D = h.shape

        # Encode current step
        current_step = self.step_encoder(h[:, -1:, :])  # (B, 1, D)

        if reasoning_buffer is None or len(reasoning_buffer) == 0:
            return current_step.squeeze(1), [current_step.detach()]

        # Attend to previous reasoning steps
        # Stack previous steps: (B, num_steps, D)
        prev_steps = torch.cat(reasoning_buffer, dim=1)

        q = self.step_query(current_step)  # (B, 1, D)
        k = self.step_key(prev_steps)      # (B, num_steps, D)
        v = self.step_value(prev_steps)    # (B, num_steps, D)

        # Attention
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D), dim=-1)
        context = attn @ v  # (B, 1, D)

        # Combine current step with context
        output = current_step + context
        output = self.output_proj(output)

        # Update buffer
        new_buffer = reasoning_buffer + [current_step.detach()]
        if len(new_buffer) > self.max_steps:
            new_buffer = new_buffer[-self.max_steps:]

        return output.squeeze(1), new_buffer


class ThinkingTokenHead(nn.Module):
    """
    Head that predicts thinking tokens for explicit reasoning.
    Extends the main head with reasoning token predictions.
    """
    def __init__(self, D, num_reasoning_tokens=4):
        super().__init__()
        self.reasoning_proj = nn.Linear(D, num_reasoning_tokens)

    def forward(self, h):
        """
        h: (B, L, D)
        Returns: (B, L, num_reasoning_tokens) logits for reasoning tokens
        """
        return self.reasoning_proj(h)
