# synthetic_trial_data

## Installation Guide

### 1. Install Required Libraries for Ubuntu/Debian

Depending on your distribution, run the appropriate set of commands:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
```

### 2. Install Python Using `pyenv`

Install `pyenv`
```bash
curl https://pyenv.run | bash
```

Set up environment for pyenv. You'll need to add pyenv to $PATH. Open your .bashrc, .zshrc, or whatever shell configuration file you use:
```bash
nano ~/.bashrc
```

After adding, you can apply the changes with:
```bash
source ~/.bashrc
```

Add the following lines to the end of the file
```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Install the desired Python version:
```bash
pyenv install 3.10.0
```

Set the newly installed Python version as global:
```bash
pyenv global 3.10.0
```

### 4. Create a Virtual Environment

You can use Python's `venv`:

```bash
python -m venv myenv
source myenv/bin/activate
```

Or use `pyenv-virtualenv` if you have it installed:

```bash
pyenv virtualenv 3.10.0 myenv
pyenv activate myenv
```

### 5. Install Dependencies with Poetry

In your project directory, run:

```bash
poetry lock
poetry install
```

### 6. Prepare computation on GPU
#### 6.1 Update System and Install Prerequisites
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential dkms
```

#### 6.2 Installing NVIDIA Drivers


