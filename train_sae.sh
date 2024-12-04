pip install --upgrade numpy
pip install datasets transformers sae-lens bitsandbytes natsort
apt install build-essential -y

python3 save_gemma_acts_and_loss.py --layers 12
./scripts/run_all.sh

python3 train_on_error.py