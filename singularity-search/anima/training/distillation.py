"""
Knowledge Distillation for ANIMA
================================

Train ANIMA using knowledge distillation from larger teacher models
(e.g., Qwen3-30B-A3B, GPT-4, etc.).

KEY TECHNIQUES:
1. Response-based distillation: Match teacher's output logits
2. Feature-based distillation: Match intermediate representations
3. Relation-based distillation: Match attention patterns / state relationships

TEACHER OPTIONS:
- Local: Qwen3-30B-A3B (3B active params, runs on RTX 4090)
- API: GPT-4, Claude, etc. (for generating training data)
- Offline: Pre-computed teacher outputs saved to disk

References:
- SVD Distillation: https://huggingface.co/BasedBase/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    # Temperature for softening logits
    temperature: float = 4.0

    # Loss weights
    alpha_ce: float = 0.5      # Cross-entropy with ground truth
    alpha_kd: float = 0.5      # KL divergence with teacher
    alpha_feature: float = 0.0  # Feature matching (optional)

    # Teacher settings
    teacher_model: str = "qwen3-30b-a3b"  # or path to saved outputs
    use_offline_teacher: bool = False

    # Training settings
    batch_size: int = 32
    max_seq_len: int = 512
    learning_rate: float = 1e-4


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    L = alpha_ce * CE(student, labels) + alpha_kd * KL(student, teacher)
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_logits: torch.Tensor,  # [B, seq, vocab]
        teacher_logits: torch.Tensor,  # [B, seq, vocab]
        labels: torch.Tensor,          # [B, seq]
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Returns dict with individual loss components and total.
        """
        T = self.config.temperature
        losses = {}

        # Cross-entropy with ground truth labels
        if self.config.alpha_ce > 0:
            # Reshape for cross entropy: [B*seq, vocab] and [B*seq]
            B, L, V = student_logits.shape
            ce = self.ce_loss(
                student_logits.view(-1, V),
                labels.view(-1)
            )
            losses['ce'] = ce

        # KL divergence with teacher (soft targets)
        if self.config.alpha_kd > 0:
            # Soften logits with temperature
            student_soft = F.log_softmax(student_logits / T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)

            # KL divergence
            kd = self.kl_loss(student_soft, teacher_soft) * (T ** 2)
            losses['kd'] = kd

        # Feature matching (optional)
        if self.config.alpha_feature > 0 and student_features is not None:
            # Project student features to teacher dimension if needed
            if student_features.shape != teacher_features.shape:
                # Simple linear projection
                proj = nn.Linear(
                    student_features.shape[-1],
                    teacher_features.shape[-1],
                    device=student_features.device
                )
                student_features = proj(student_features)

            feature_loss = F.mse_loss(student_features, teacher_features)
            losses['feature'] = feature_loss

        # Combined loss
        total = (
            self.config.alpha_ce * losses.get('ce', 0) +
            self.config.alpha_kd * losses.get('kd', 0) +
            self.config.alpha_feature * losses.get('feature', 0)
        )
        losses['total'] = total

        return losses


class OfflineTeacherDataset(torch.utils.data.Dataset):
    """
    Dataset for offline distillation with pre-computed teacher outputs.

    Expects JSONL files with format:
    {"input": [...], "label": [...], "teacher_logits": [...]}
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_seq_len: int = 512,
    ):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len

        # Load data
        self.samples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors
        input_ids = torch.tensor(sample['input'][:self.max_seq_len])
        labels = torch.tensor(sample['label'][:self.max_seq_len])

        # Teacher logits (may be top-k only for memory efficiency)
        if 'teacher_logits' in sample:
            teacher_logits = torch.tensor(sample['teacher_logits'])
        else:
            teacher_logits = None

        return {
            'input_ids': input_ids,
            'labels': labels,
            'teacher_logits': teacher_logits,
        }


class OnlineTeacher(nn.Module):
    """
    Wrapper for online teacher model inference.

    Supports:
    - Local models via transformers (Qwen3, Llama, etc.)
    - Caching of teacher outputs for efficiency
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load teacher model (lazy loading to save memory)."""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading teacher model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"Teacher model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get teacher logits for input."""
        self.load_model()

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        return outputs.logits

    def generate_distillation_data(
        self,
        texts: List[str],
        output_path: Path,
        batch_size: int = 4,
    ):
        """
        Generate offline distillation data from teacher.

        Saves teacher logits (top-k for memory efficiency) to JSONL.
        """
        self.load_model()
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                # Get teacher logits
                logits = self.get_logits(
                    encoded['input_ids'],
                    encoded['attention_mask'],
                )

                # Save top-k logits for memory efficiency
                top_k = 100
                top_logits, top_indices = logits.topk(top_k, dim=-1)

                # Write samples
                for j, text in enumerate(batch_texts):
                    sample = {
                        'text': text,
                        'input': encoded['input_ids'][j].tolist(),
                        'label': encoded['input_ids'][j].tolist(),  # For LM
                        'teacher_top_logits': top_logits[j].cpu().tolist(),
                        'teacher_top_indices': top_indices[j].cpu().tolist(),
                    }
                    f.write(json.dumps(sample) + '\n')

                if (i // batch_size) % 10 == 0:
                    print(f"Processed {i + len(batch_texts)}/{len(texts)} samples")


class ANIMADistillationTrainer:
    """
    Trainer for distilling knowledge into ANIMA.

    Supports both online and offline teacher modes.
    """

    def __init__(
        self,
        student_model: nn.Module,
        config: DistillationConfig,
        teacher: Optional[OnlineTeacher] = None,
    ):
        self.student = student_model
        self.config = config
        self.teacher = teacher
        self.loss_fn = DistillationLoss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, float]:
        """Single training step."""
        self.student.train()

        # Move to device
        input_data = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Student forward
        student_output = self.student(input_data.unsqueeze(-1).float())  # Add feature dim

        # Get teacher logits (online or offline)
        if batch.get('teacher_logits') is not None:
            teacher_logits = batch['teacher_logits'].to(device)
        elif self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher.get_logits(input_data)
        else:
            # Self-distillation or no teacher
            teacher_logits = student_output.detach()

        # Compute loss
        losses = self.loss_fn(
            student_logits=student_output,
            teacher_logits=teacher_logits,
            labels=labels,
        )

        # Backward
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {}

        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_step(batch, device)

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}: loss={losses['total']:.4f}")

        # Average
        n_batches = len(dataloader)
        return {k: v / n_batches for k, v in epoch_losses.items()}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_synthetic_teacher_data(
    num_samples: int = 1000,
    seq_len: int = 64,
    vocab_size: int = 100,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Create synthetic teacher data for testing distillation.

    In production, use OnlineTeacher.generate_distillation_data()
    with a real teacher model.
    """
    samples = []

    for i in range(num_samples):
        # Random input sequence
        input_ids = torch.randint(0, vocab_size, (seq_len,))

        # Synthetic "teacher" logits (random but structured)
        teacher_logits = torch.randn(seq_len, vocab_size)
        # Make it peaky to simulate confident teacher
        teacher_logits = F.softmax(teacher_logits * 2.0, dim=-1)

        sample = {
            'input': input_ids.tolist(),
            'label': input_ids.tolist(),  # Next token prediction
            'teacher_logits': teacher_logits.tolist(),
        }
        samples.append(sample)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

    return samples


def quick_distillation_test():
    """Quick test of distillation pipeline."""
    from anima.core import AnimaOptimized

    print("=" * 60)
    print("DISTILLATION PIPELINE TEST")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create student model
    student = AnimaOptimized(
        sensory_dim=1,  # Single token embedding
        d_model=32,
        bottleneck_dim=16,
        output_dim=100,  # Vocab size
    ).to(device)

    print(f"Student params: {student.count_parameters():,}")

    # Create config
    config = DistillationConfig(
        temperature=4.0,
        alpha_ce=0.5,
        alpha_kd=0.5,
        use_offline_teacher=True,
    )

    # Create synthetic teacher data
    print("\nGenerating synthetic teacher data...")
    samples = create_synthetic_teacher_data(num_samples=100, seq_len=32, vocab_size=100)

    # Create dataset and dataloader
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            return {
                'input_ids': torch.tensor(s['input']),
                'labels': torch.tensor(s['label']),
                'teacher_logits': torch.tensor(s['teacher_logits']),
            }

    dataset = SimpleDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Create trainer
    trainer = ANIMADistillationTrainer(student, config)

    # Train for a few epochs
    print("\nTraining with distillation...")
    for epoch in range(3):
        losses = trainer.train_epoch(dataloader, device)
        print(f"Epoch {epoch+1}: loss={losses['total']:.4f}, ce={losses.get('ce', 0):.4f}, kd={losses.get('kd', 0):.4f}")

    print("\nDistillation test complete!")


if __name__ == "__main__":
    quick_distillation_test()
