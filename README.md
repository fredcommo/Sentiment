## Run multi-models sentiment analysis on verbatims

## Install python environment
Either run
```bash
./install_env.sh
```
or manually run the following steps
```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

python -m venv sentiment_env
source sentiment_env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m textblob.download_corpora

# optional, in case you need a notebook kernel
python -m ipykernel install --user --name sentiment_env --display-name "Python3.x (sentiment)"
```
