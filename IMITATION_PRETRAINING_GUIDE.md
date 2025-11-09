# Imitation Pre-training + RL Fine-tuning Guide

Speed up RL training by starting with an imitation learning baseline instead of random initialization.

## Why Use Imitation Pre-training?

**Problem:** RL from scratch is slow because the robot starts with completely random movements.

**Solution:** Pre-train with imitation learning to give the robot a reasonable starting policy.

### Benefits

1. **Faster Convergence** - RL training converges 2-10x faster
2. **Better Final Performance** - Often achieves higher rewards
3. **More Stable** - Reduces early training instability
4. **Sample Efficient** - Requires fewer environment interactions

## Quick Start

### Option 1: Full Pipeline (Recommended for First Time)

Train everything from scratch:

```bash
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --n-demo-episodes 50 \
    --total-bc-epochs 200 \
    --total-rl-timesteps 500000 \
    --visualize-freq 50000
```

This will:
1. Generate 50 demonstration episodes with hardcoded gait
2. Pre-train a BC policy for 200 epochs
3. Fine-tune with RL for 500K timesteps
4. Show visualization every 50K steps

### Option 2: Separate Steps (More Control)

**Step 1:** Generate demonstrations and train BC

```bash
# This will create BC policy and save demonstrations
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --n-demo-episodes 100 \
    --total-bc-epochs 300 \
    --total-rl-timesteps 0  # Skip RL for now
```

**Step 2:** Fine-tune with RL using the BC policy

```bash
# Use the BC policy from previous step
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --pretrained-model outputs/bc_rl_TIMESTAMP/bc_pretrained_policy.pth \
    --skip-bc \
    --total-rl-timesteps 1000000
```

## How It Works

### Phase 1: Demonstration Collection

The script generates demonstrations using a **hardcoded walking gait**:

```python
# Diagonal trot gait (FL & RR together, FR & RL together)
action = np.array([
    0.2 * np.sin(t) - 0.1,        # FL hip
    0.4 * np.cos(t) - 0.2,        # FL knee
    0.2 * np.sin(t + π) - 0.1,    # FR hip
    0.4 * np.cos(t + π) - 0.2,    # FR knee
    # ... rear legs
])
```

This creates a simple but functional walking motion that the robot can learn from.

### Phase 2: Behavioral Cloning (BC)

BC learns to imitate the demonstrations using supervised learning:

- **Input:** Robot observations (joint positions, velocities, IMU, etc.)
- **Output:** Actions (target joint positions)
- **Loss:** Mean squared error between predicted and demonstrated actions

BC trains quickly (minutes) and creates a reasonable baseline policy.

### Phase 3: RL Fine-tuning

PPO is initialized with BC weights and fine-tuned with RL:

- Starts from BC policy (not random)
- Optimizes for actual rewards (forward velocity, stability, efficiency)
- Improves beyond the demonstrations

## Command-Line Options

### Demonstration Collection

```bash
--n-demo-episodes N       # Number of demo episodes to collect (default: 50)
                          # More demos = better BC, but slower
                          # 50-100 is usually sufficient
```

### BC Pre-training

```bash
--total-bc-epochs N       # BC training epochs (default: 200)
                          # More epochs = better BC policy
                          # 200-500 works well

--skip-bc                 # Skip BC training (if you have pretrained model)

--pretrained-model PATH   # Use existing BC model instead of training new one
```

### RL Fine-tuning

```bash
--total-rl-timesteps N    # RL training timesteps (default: 500000)
                          # Usually need less than training from scratch

--n-envs N                # Parallel environments (default: 4)

--visualize-freq N        # Visualize every N timesteps (default: 50000)
                          # Set to 0 to disable

--learning-rate F         # Learning rate (default: 3e-4)
```

### General Options

```bash
--use-camera              # Use camera observations (slower but more powerful)

--output-dir PATH         # Output directory (default: outputs)

--experiment-name NAME    # Experiment name (default: auto-generated)

--seed N                  # Random seed (default: 42)
```

## Examples

### Example 1: Quick Test (Minimal Training)

```bash
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --n-demo-episodes 20 \
    --total-bc-epochs 50 \
    --total-rl-timesteps 100000 \
    --visualize-freq 25000
```

Fast pipeline for testing (10-15 minutes total).

### Example 2: Production Training (High Quality)

```bash
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --n-demo-episodes 100 \
    --total-bc-epochs 500 \
    --total-rl-timesteps 2000000 \
    --n-envs 8 \
    --visualize-freq 100000
```

High-quality training with more demonstrations and longer RL fine-tuning.

### Example 3: Resume from Checkpoint

```bash
# First, do BC pre-training
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --n-demo-episodes 100 \
    --total-bc-epochs 300 \
    --total-rl-timesteps 0 \
    --experiment-name my_bc_model

# Later, fine-tune with RL
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --pretrained-model outputs/my_bc_model/bc_pretrained_policy.pth \
    --skip-bc \
    --total-rl-timesteps 1000000 \
    --experiment-name my_finetuned_model
```

### Example 4: With Camera Observations

```bash
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --use-camera \
    --n-demo-episodes 100 \
    --total-bc-epochs 400 \
    --total-rl-timesteps 1000000 \
    --n-envs 4
```

Uses visual observations (slower but potentially more capable).

## Expected Timeline

### Without Pre-training (Random Init)
- **0-50K steps:** Random thrashing, falling
- **50K-200K steps:** Starting to stand
- **200K-500K steps:** Beginning to walk
- **500K-1M steps:** Decent walking
- **Total:** ~1-2 million steps needed

### With Pre-training (BC Init)
- **0 steps:** Already walking (from BC)
- **0-100K steps:** Refining gait
- **100K-300K steps:** Optimizing speed
- **300K-500K steps:** Near-optimal walking
- **Total:** ~300K-500K steps needed

**Speedup:** ~3-4x faster convergence!

## Workflow Diagram

```
┌─────────────────────────────────────┐
│ 1. Hardcoded Demonstrations         │
│    - Generate 50-100 episodes       │
│    - Simple trot gait pattern       │
│    - Saves to demonstrations.pkl    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. Behavioral Cloning (BC)          │
│    - Train for 200-500 epochs       │
│    - Supervised learning (MSE loss) │
│    - Fast training (~5-10 minutes)  │
│    - Saves BC policy weights        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Initialize PPO with BC Weights   │
│    - Copy BC policy → PPO policy    │
│    - Start with functional behavior │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. RL Fine-tuning (PPO)             │
│    - Optimize for actual rewards    │
│    - Improve beyond demonstrations  │
│    - Faster convergence             │
│    - Better final performance       │
└─────────────────────────────────────┘
```

## Comparing Approaches

| Approach | Time to Walk | Final Quality | Sample Efficiency |
|----------|--------------|---------------|-------------------|
| **RL from Scratch** | ~500K steps | Good | Low |
| **BC Only** | Immediate | Limited | Very High |
| **BC → RL** | ~100K steps | Excellent | High |

**Recommendation:** Use BC → RL for best results!

## Tips for Best Results

### 1. Tune Demonstration Quality

More demonstrations = better BC baseline:
- 20 episodes: Minimal baseline
- 50 episodes: Good baseline (recommended)
- 100+ episodes: Excellent baseline

### 2. BC Training Duration

- Too few epochs: Underfitting, poor baseline
- Just right: 200-500 epochs
- Too many epochs: Overfitting (less flexible for RL)

### 3. RL Fine-tuning

Start with shorter RL training when using BC:
- BC already provides good policy
- 300K-500K steps often sufficient
- Can always train longer if needed

### 4. Learning Rate

May want to use lower learning rate for fine-tuning:
```bash
--learning-rate 1e-4  # Lower than default 3e-4
```

This preserves BC knowledge while allowing refinement.

## Troubleshooting

### BC policy doesn't walk well

**Solution:** Collect more demonstrations or train longer
```bash
--n-demo-episodes 100 --total-bc-epochs 500
```

### RL forgets BC knowledge

**Solution:** Use lower learning rate
```bash
--learning-rate 1e-4
```

### Training still slow

**Solution:** Use more parallel environments
```bash
--n-envs 8
```

### Demonstrations not realistic

The hardcoded gait is intentionally simple. For better demos:
1. Train a model with pure RL first
2. Use that model to generate demonstrations
3. Use those demos for BC pre-training on new models

## Advanced: Using Your Own Expert

If you have a well-trained model, use it to generate better demonstrations:

```python
# Generate demonstrations from trained model
from imitation.data import rollout
expert_model = PPO.load("path/to/expert/model.zip")
env = DummyVecEnv([lambda: PiDogEnv()])
transitions = rollout.rollout(
    expert_model,
    env,
    rollout.make_sample_until(min_episodes=100),
)

# Save for BC training
with open("expert_demos.pkl", "wb") as f:
    pickle.dump(transitions, f)
```

Then use with the pipeline:
```bash
# Modify train_pretrain_finetune.py to load your expert_demos.pkl
```

## See Also

- `train_pretrain_finetune.py` - Full BC → RL pipeline
- `training/train_imitation.py` - Standalone BC/GAIL training
- `training/train_rl.py` - Standard RL training (no pre-training)
- `VISUALIZATION_GUIDE.md` - Real-time visualization
