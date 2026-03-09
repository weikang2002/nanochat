"""
Minimal, educational REINFORCE-style RL fine-tuning on GSM8K.

This file is a simplified, single-device rewrite of chat_rl.py for clarity.
It strips away distributed training, wandb, and other production concerns so
the core algorithm is as visible as possible.

The algorithm
─────────────
This is REINFORCE with a mean-baseline advantage, also known as a
simplified variant of GRPO (Group Relative Policy Optimization):

  For each question q:
    1. Sample G completions {a₁, …, a_G} from the current policy π_θ
    2. Score each completion: rᵢ = reward(q, aᵢ)   ← binary 0/1 for GSM8K
    3. Compute advantage:     Aᵢ = rᵢ − mean(r)     ← center rewards around 0
    4. Policy gradient loss:  L = −∑_{i,t} log π_θ(aᵢₜ | context) · Aᵢ
    5. Gradient step on L.

Why subtract the mean?
  Without a baseline every correct answer would push gradients up, but so
  would the *prompt* tokens — causing instability.  Subtracting mean(r)
  turns the signal into "was this completion better than average?", which
  is a much cleaner learning signal.

Differences vs full GRPO / PPO:
  - No KL penalty / reference model                (trust-region removed)
  - No importance ratio+clip                       (on-policy, no need)
  - No ÷ σ in the advantage                        (DAPO-style, just − μ)

Usage (MacBook CPU or Apple Silicon):
  python -m scripts.rl_simple --max-steps 20 --num-samples 4

GPU (single card):
  python -m scripts.rl_simple --device-type cuda
"""

import argparse
import os
import itertools
import torch
import torch.nn.functional as F

from nanochat.common import get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simple educational RL on GSM8K")
parser.add_argument("--device-type",   type=str,   default="",        help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag",     type=str,   default=None,      help="SFT checkpoint tag to start from")
parser.add_argument("--model-step",    type=int,   default=None,      help="SFT checkpoint step to load")
parser.add_argument("--max-steps",     type=int,   default=500,       help="total optimisation steps")
# Rollout settings
parser.add_argument("--num-samples",   type=int,   default=8,         help="completions sampled per question (G)")
parser.add_argument("--max-new-tokens",type=int,   default=256,       help="max generated tokens per completion")
parser.add_argument("--temperature",   type=float, default=1.0,       help="sampling temperature")
parser.add_argument("--top-k",         type=int,   default=50,        help="top-k sampling (0 = off)")
# Optimizer
parser.add_argument("--lr",            type=float, default=1e-5,      help="flat learning rate (simple override)")
# Evaluation / checkpointing
parser.add_argument("--eval-every",    type=int,   default=50,        help="evaluate greedy accuracy every N steps")
parser.add_argument("--eval-examples", type=int,   default=100,       help="number of val examples for quick eval")
parser.add_argument("--save-every",    type=int,   default=100,       help="save checkpoint every N steps")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Device setup  (no DDP — single process, works on MacBook MPS or CPU)
# ─────────────────────────────────────────────────────────────────────────────
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Model, tokenizer, inference engine
# ─────────────────────────────────────────────────────────────────────────────
model, tokenizer, meta = load_model(
    "sft", device, phase="train",
    model_tag=args.model_tag, step=args.model_step
)
engine = Engine(model, tokenizer)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
train_task = GSM8K(subset="main", split="train")
val_task   = GSM8K(subset="main", split="test")
print(f"Train size: {len(train_task)} | Val size: {len(val_task)}")

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer  (simple Adam over all parameters for clarity)
# ─────────────────────────────────────────────────────────────────────────────
# In production code (chat_rl.py) a mixed Adam+Muon optimizer is used with
# separate learning rates per parameter group.  Here we use a single Adam
# for simplicity so the training loop is easier to follow.
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  — Rollout: sample G completions for one question
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_rollouts(conversation: dict, step: int) -> dict:
    """
    Given a single training example (question + ground-truth answer), generate
    `args.num_samples` completions from the current policy and score them.

    Returns a dict with everything needed to compute the policy gradient:
        completions  : list of full token sequences  (len = G)
        masks        : list of binary masks          (len = G)  1 = train on this token
        prefix_len   : int  length of the prompt tokens (same for all completions)
        rewards      : 1-D float tensor of shape (G,)
        advantages   : 1-D float tensor of shape (G,)  = rewards − mean(rewards)
    """
    model.eval()   # sampling must always be done in eval mode

    # Tokenize the prompt (user turn + empty assistant turn to prime generation)
    prompt_tokens = tokenizer.render_for_completion(conversation)
    prefix_len    = len(prompt_tokens)

    # Sample G completions.  engine.generate_batch returns:
    #   completions[i]  : full token sequence (prompt + generated)    list[int]
    #   masks[i]        : 1 for generated tokens, 0 for prompt tokens  list[int]
    seed = hash((step,)) & 0x7FFFFFFF
    completions, masks = engine.generate_batch(
        prompt_tokens,
        num_samples=args.num_samples,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=seed,
    )

    # ── STEP 2: score each completion ────────────────────────────────────────
    # The reward is binary: 1 if the model extracted the correct final number,
    # 0 otherwise.  No learned reward model is involved.
    rewards = []
    for tokens_seq in completions:
        generated_text = tokenizer.decode(tokens_seq[prefix_len:])
        r = train_task.reward(conversation, generated_text)   # 0.0 or 1.0
        rewards.append(r)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    # ── STEP 3: compute advantages ───────────────────────────────────────────
    # Advantage = reward − mean(reward)
    #
    # Intuition: if all completions score 0 (all wrong) the mean is 0 and
    # advantages are all 0 → no gradient signal, which is correct because we
    # don't know *how* to improve.  When some are right and some wrong, the
    # correct ones get positive advantage (+) and the wrong ones negative (−).
    advantages = rewards - rewards.mean()

    return dict(
        completions=completions,
        masks=masks,
        prefix_len=prefix_len,
        rewards=rewards,
        advantages=advantages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  — Policy gradient update
# ─────────────────────────────────────────────────────────────────────────────
def policy_gradient_step(rollout: dict) -> float:
    """
    Given a completed rollout dict (from collect_rollouts), run one forward
    pass and accumulate gradients.

    The policy gradient (REINFORCE) objective is:
        J(θ) = E[ A · log π_θ(a | s) ]

    We maximize J, i.e. minimize L = −J.

    The model's forward() returns per-token NLL (negative log-likelihood),
    so  log π_θ = −NLL  and we can write:

        L = (1/N) ∑_{i,t}  NLL(aᵢₜ) · (−Aᵢ)          [N = #valid tokens]

    Tokens that belong to the prompt or padding have mask=0 and are excluded
    via the ignore_index=-1 mechanism in CrossEntropyLoss.
    """
    model.train()  # enable dropout / gradient tracking

    completions = rollout["completions"]
    masks       = rollout["masks"]
    advantages  = rollout["advantages"]   # shape (G,)

    # ── Build padded input / target tensors ──────────────────────────────────
    pad_token   = tokenizer.encode_special("<|assistant_end|>")
    max_len     = max(len(s) for s in completions)

    padded = [s + [pad_token] * (max_len - len(s)) for s in completions]
    padded_masks = [m + [0] * (max_len - len(m))   for m in masks]

    ids      = torch.tensor(padded,        dtype=torch.long,  device=device)  # (G, T)
    mask_ids = torch.tensor(padded_masks,  dtype=torch.long,  device=device)  # (G, T)

    # Autoregressive: input is everything except the last token,
    # target is everything except the first token (shifted by 1).
    inputs  = ids[:, :-1]                          # (G, T-1)
    targets = ids[:, 1:].clone()                   # (G, T-1)
    # Mask out prompt tokens and padding so they do not contribute to the loss.
    # CrossEntropyLoss with ignore_index=-1 will silently skip those positions.
    targets[mask_ids[:, 1:] == 0] = -1

    # ── Forward pass ─────────────────────────────────────────────────────────
    # Internally the model calls targets.view(-1) before F.cross_entropy, so
    # with reduction='none' it returns a flat 1-D tensor of length G*(T-1).
    # view_as(inputs) reshapes it back to (G, T-1) and negating NLL gives logp.
    nll  = model(inputs, targets, loss_reduction='none')  # (G*(T-1),) flat
    logp = -nll.view_as(inputs)                           # (G, T-1)

    # ── Policy gradient objective ─────────────────────────────────────────────
    # L = −(1/N) ∑_{i,t}  logp[i,t] · A[i]
    #
    # advantages[:, None] broadcasts (G,) → (G, T-1) so each token of
    # completion i is weighted by that completion's advantage.
    pg_obj   = (logp * advantages.unsqueeze(-1)).sum()
    n_valid  = (targets >= 0).sum().clamp(min=1)   # number of non-masked tokens
    pg_obj   = pg_obj / n_valid

    loss = -pg_obj   # flip sign: we minimize, not maximize
    loss.backward()

    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Quick greedy evaluation (Pass@1)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(max_examples: int = 100) -> float:
    """Greedy decoding accuracy on the first `max_examples` val problems."""
    model.eval()
    correct = 0
    for idx in range(min(max_examples, len(val_task))):
        conversation  = val_task[idx]
        prompt_tokens = tokenizer.render_for_completion(conversation)
        prefix_len    = len(prompt_tokens)
        completions, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=0.0,   # greedy
            top_k=0,
        )
        text = tokenizer.decode(completions[0][prefix_len:])
        correct += val_task.evaluate(conversation, text)
    accuracy = correct / min(max_examples, len(val_task))
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  REINFORCE on GSM8K  |  steps={args.max_steps}  G={args.num_samples}")
print(f"{'─'*60}\n")

example_iter = itertools.cycle(range(len(train_task)))   # infinite data loop

for step in range(args.max_steps):

    # ── Evaluation ───────────────────────────────────────────────────────────
    if step % args.eval_every == 0:
        acc = evaluate(args.eval_examples)
        print(f"[step {step:4d}]  Val greedy accuracy (Pass@1): {acc:.3f}")

    # ── Collect rollout ───────────────────────────────────────────────────────
    example_idx  = next(example_iter)
    conversation = train_task[example_idx]
    rollout      = collect_rollouts(conversation, step)

    mean_r  = rollout["rewards"].mean().item()
    nonzero = (rollout["rewards"] > 0).sum().item()

    # ── Policy gradient update ────────────────────────────────────────────────
    optimizer.zero_grad()
    loss = policy_gradient_step(rollout)
    optimizer.step()

    print(
        f"[step {step:4d}]  loss={loss:+.4f}  "
        f"mean_reward={mean_r:.3f}  correct={nonzero}/{args.num_samples}"
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    is_last = step == args.max_steps - 1
    if (step > 0 and step % args.save_every == 0) or is_last:
        base_dir      = get_base_dir()
        depth         = model.config.n_layer
        tag           = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", tag)
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            optimizer_data=None,   # skip optimizer state for simplicity
            meta_data={"model_config": model.config.__dict__},
        )
        print(f"  ✓ Saved checkpoint → {checkpoint_dir}")

print("\nDone.")
