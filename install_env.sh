# Install Rust compiler with:
# curl https://sh.rustup.rs -sSf | sh -s -- -y
# export PATH="$HOME/.cargo/bin:$PATH"

# Install Rust compiler with:
curl https://sh.rustup.rs -sSf | sh -s -- -y

PATTERN="/.cargo/bin"
if $(echo $PATH | grep -q $PATTERN); then
    echo "$PATTERN already in path"
else
    export PATH="$HOME/.cargo/bin:$PATH"
fi

python -m pip install --upgrade pip

# Define venv name
VENV_NAME="sentiment_env"
python -m venv $VENV_NAME
source $VENV_NAME/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m textblob.download_corpora

# Install ipykernel
# The kernel will be available in notebooks as: python <x.y.z> (<VENV_NAME>)
PYTHON_VERSION=$(python -V)
DISPLAY_NAME="$PYTHON_VERSION ($VENV_NAME)"

echo "Installing ipykernel for:"
echo "      - environment name = $VENV_NAME"
echo "      - python version = $PYTHON_VERSION"
echo "      - display-name = $DISPLAY_NAME"
python -m ipykernel install --user --name $VENV_NAME --display-name "$DISPLAY_NAME"
