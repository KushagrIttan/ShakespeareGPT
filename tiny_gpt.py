"""A small, readable character-level GPT. No pretrained weights or transformer wrappers."""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import time
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


@dataclass
class Config:
    context: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    steps: int = 1000
    learning_rate: float = 3e-4
    eval_every: int = 100
    eval_batches: int = 10
    seed: int = 1337


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.width % cfg.heads == 0
        self.heads = cfg.heads
        self.head_width = cfg.width // cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.projection = nn.Linear(cfg.width, cfg.width)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(cfg.context, cfg.context, dtype=torch.bool)))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # Each head works with its own smaller representation of each character.
        q, k, v = [t.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
                   for t in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        # A position can inspect itself and its past, but never future characters.
        scores = scores.masked_fill(~self.mask[:length, :length], float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        mixed = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.dropout(self.projection(mixed))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attention = nn.LayerNorm(cfg.width)
        self.attention = CausalAttention(cfg)
        self.norm_mlp = nn.LayerNorm(cfg.width)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(),
                                 nn.Linear(4 * cfg.width, cfg.width), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(vocab_size, cfg.width)
        self.positions = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.output = nn.Linear(cfg.width, vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        length = ids.shape[1]
        if length > self.cfg.context:
            raise ValueError('Input exceeds the context window.')
        positions = torch.arange(length, device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        logits = self.output(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, count=300, temperature=0.8, top_k=None):
        if temperature <= 0:
            raise ValueError('Temperature must be positive.')
        if top_k is not None and top_k <= 0:
            raise ValueError('top_k must be positive.')
        was_training = self.training
        self.eval()
        try:
            for _ in range(count):
                logits, _ = self(ids[:, -self.cfg.context:])
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                ids = torch.cat((ids, next_id), dim=1)
            return ids
        finally:
            self.train(was_training)


def read_data(path, cfg):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    text = path.read_text(encoding='utf-8')
    chars = sorted(set(text))
    stoi = {char: i for i, char in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    boundary = int(0.9 * len(ids))
    splits = {'train': ids[:boundary], 'val': ids[boundary:]}
    if min(len(part) for part in splits.values()) <= cfg.context:
        raise ValueError('Both text splits must be longer than the context window.')
    return chars, stoi, splits


def get_batch(data, cfg, device, generator=None):
    starts = torch.randint(len(data) - cfg.context, (cfg.batch_size,), generator=generator)
    x = torch.stack([data[i:i + cfg.context] for i in starts])
    y = torch.stack([data[i + 1:i + cfg.context + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, splits, cfg, device):
    was_training = model.training
    model.eval()
    # Reuse the same held-out windows at each evaluation for a fair comparison.
    generator = torch.Generator().manual_seed(2026)
    result = {}
    for name, data in splits.items():
        losses = []
        for _ in range(cfg.eval_batches):
            x, y = get_batch(data, cfg, device, generator)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train(was_training)
    return result


def train(cfg, data_path, output_dir, device='auto'):
    if min(cfg.steps, cfg.eval_every, cfg.eval_batches, cfg.batch_size, cfg.context,
           cfg.width, cfg.heads, cfg.layers) < 1:
        raise ValueError('Counts and model dimensions must be positive.')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)
    if device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chars, stoi, splits = read_data(data_path, cfg)
    model = TinyGPT(len(chars), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'Device: {device} | characters: {len(chars)} | parameters: {parameters:,}', flush=True)
    prompt = 'ROMEO:\n' if all(c in stoi for c in 'ROMEO:\n') else chars[0]
    start = torch.tensor([[stoi[c] for c in prompt]], device=device)
    # Isolate sampling so collecting a sample does not change the training RNG.
    def sample():
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith('cuda') else []):
            torch.manual_seed(42)
            ids = model.generate(start.clone(), count=400)[0].tolist()
        return ''.join(chars[i] for i in ids)
    before = sample()
    history = []
    started = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            losses = evaluate(model, splits, cfg, device)
            history.append({'step': step, **losses})
            print(f"step {step:4d} | train {losses['train']:.3f} | val {losses['val']:.3f}", flush=True)
        if step == cfg.steps:
            break
        model.train()
        x, y = get_batch(splits['train'], cfg, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - started
    after = sample()
    torch.save({'config': asdict(cfg), 'chars': chars, 'model': model.state_dict()}, output_dir / 'model.pt')
    metrics = {'config': asdict(cfg), 'device': device, 'parameters': parameters,
               'seconds': elapsed, 'history': history}
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (output_dir / 'samples.txt').write_text('BEFORE TRAINING\n' + before + '\n\nAFTER TRAINING\n' + after,
                                           encoding='utf-8')
    print(f'Finished in {elapsed:.1f}s. Saved checkpoint, metrics, and samples to {output_dir}', flush=True)
    print('\nAFTER TRAINING\n' + after, flush=True)
    return model, chars, history


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TinyGPT(len(checkpoint['chars']), Config(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint['chars']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/input.txt')
    parser.add_argument('--out', default='runs/first')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    defaults = Config()
    for name in ('steps', 'batch_size', 'context', 'width', 'heads', 'layers', 'eval_every', 'eval_batches'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(defaults, name))
    args = parser.parse_args()
    cfg = Config(**{name: getattr(args, name) for name in
                    ('steps', 'batch_size', 'context', 'width', 'heads', 'layers', 'eval_every', 'eval_batches')})
    train(cfg, args.data, args.out, args.device)


if __name__ == '__main__':
    main()
