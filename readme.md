# Machine Translation (EN to DE) using PyTorch with a manually-coded Transformer

This repository contains the code for a machine translation model that translates text from English to German. The model is implemented using PyTorch and trained on a subset of the Europarl Parallel Corpus.

## Project Structure

The project is structured as follows:

- `europarl-v7.en_20k_lines.txt`: The English text file containing 20,000 lines from the Europarl Parallel Corpus.
- `europarl-v7.de_20k_lines.txt`: The corresponding German text file.
- `europarl-v7.en_tiny_lines.txt`: A small set of English validation lines.
- `europarl-v7.de_tiny_lines.txt`: A small set of German validation lines.
- `test-en.txt`: Test file containing English sentences to be translated.
- `transformer_best.pth`: The saved model checkpoint after training.

## Setup

### Google Colab/Local Setup

To set up the project, follow these steps:

1. Mount your Google Drive to access the dataset and save models:

```python
from google.colab import drive
drive.mount('/content/drive')
Install the required packages:
!pip install cloud-tpu-client
!pip install torch
!pip install torchvision
!pip install torch-xla
!pip install spacy
!python -m spacy download de_core_news_sm
!python -m spacy download en_core_web_sm
!pip install --upgrade torchtext
Data Preprocessing
Use the load_data function to load the text files and preprocess the data:

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data.dataset import random_split
```


# Load the data
```python
english_sentences, german_sentences = load_data(en_file, de_file)
```
Model Training
To train the model, run the training loop and save the best model checkpoint:

# Model training code here
The best model will be saved to transformer_best.pth.

Model Evaluation
Evaluate the model using the BLEU score:
```python
from nltk.translate.bleu_score import corpus_bleu
```
# Calculate BLEU score
```python
bleu_score = corpus_bleu(...)
print(f'BLEU score on validation data: {bleu_score:.4f}')
```
Translation
To translate the test set and print the translations:

# Translate the test set
```python
translate_and_print_test_set(...)
```
Usage
To use the trained model for translation, load the model checkpoint and call the translation function:

# Load the best model state
```python
checkpoint = torch.load(best_model_path)
```
# Translate sentences
```python
translations = translate_and_print_test_set(...)
```

Contributions
Contributions to this project are welcome. Please fork the repository and submit a pull request.

License
This project is open-sourced under the Apache v2.0 License. See the LICENSE file for more information.

