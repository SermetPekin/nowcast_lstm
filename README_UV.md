# nowcast_lstm with UV

Fast installation guide for [UV](https://github.com/astral-sh/uv) - a modern Python package installer (10-100x faster than pip).

## Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip/Homebrew
pip install uv
brew install uv  # macOS only
```

## Quick Install

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch (choose your platform)
# macOS/Linux:
uv pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Windows:
uv pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install nowcast-lstm
uv pip install nowcast-lstm
```

## Install from Source

```bash
git clone https://github.com/dhopp1/nowcast_lstm.git
cd nowcast_lstm
uv venv
source .venv/bin/activate
uv pip install -e .

# Install with optional dependencies for examples and development
uv pip install -e ".[examples,dev]"

# Or install specific extras
uv pip install -e ".[examples]"  # Just example dependencies (matplotlib, etc.)
uv pip install -e ".[dev]"       # Development dependencies (pytest, examples, etc.)
```

## Usage

Same as pip installation - see main [README.md](README.md) for full documentation.

```python
from nowcast_lstm.LSTM import LSTM

model = LSTM(data, "target_col_name", n_timesteps=12)
model.train()
predictions = model.predict(model.data)
```

## Common Commands

| Task | Command |
|------|---------|
| Create venv | `uv venv` |
| Install package | `uv pip install package-name` |
| Install from requirements | `uv pip install -r requirements.txt` |
| Editable install | `uv pip install -e .` |
| Editable install with extras | `uv pip install -e ".[examples,dev]"` |
| List packages | `uv pip list` |

## GPU Support

```bash
# CUDA 11.1
uv pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
uv pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Troubleshooting

**UV not found:** Add to PATH in `~/.bashrc` or `~/.zshrc`:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

**PyTorch issues:** Install separately first:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install nowcast-lstm
```

**Dependency conflicts:**
```bash
uv cache clean
uv pip install -v nowcast-lstm
```

## Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [Main README](README.md)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
