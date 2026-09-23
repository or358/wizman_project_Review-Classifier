# deep_learning_methods/main.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from gensim.models import Word2Vec
import torchtext.vocab as vocab_lib

from config import ModelConfig, Word2VecConfig
from preprocessor import TextPreprocessor
from data_loader import Vocabulary, TextDataset
from models import TextCNN, TextLSTM
from trainer import ModelTrainer
from transformer_main import train_transformer  # Importing the transformer runner

def load_pretrained_glove(vocab: Vocabulary):
    """
    Loads pretrained GloVe embeddings for the words in our vocabulary.
    Returns an embedding matrix matching our vocabulary size.
    """
    print("\nDownloading/Loading GloVe pretrained vectors (this might take a minute the first time)...")
    # Using GloVe 6B tokens, 300 dimensions as a standard pretrained model
    glove = vocab_lib.GloVe(name='6B', dim=300)
    
    embedding_matrix = torch.zeros((vocab.vocab_size, 300))
    nn.init.uniform_(embedding_matrix, -0.1, 0.1)
    embedding_matrix[vocab.word2idx[ModelConfig.PAD_TOKEN]] = torch.zeros(300)
    
    words_found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove.stoi:
            embedding_matrix[idx] = glove.vectors[glove.stoi[word]]
            words_found += 1
            
    print(f"Loaded {words_found}/{vocab.vocab_size} pretrained GloVe vectors.")
    return embedding_matrix, 300

def create_our_w2v_matrix(vocab: Vocabulary, w2v_path: str = "/data/my_word2vec.model"):
    """
    Loads our custom trained Word2Vec model.
    """
    if not os.path.exists(w2v_path):
        raise FileNotFoundError(f"Cannot find Word2Vec model at {w2v_path}. Did you run train_word2vec.py?")
        
    print(f"\nLoading custom trained Word2Vec model from {w2v_path}...")
    w2v_model = Word2Vec.load(w2v_path)
    
    embedding_matrix = torch.zeros((vocab.vocab_size, Word2VecConfig.VECTOR_SIZE))
    nn.init.uniform_(embedding_matrix, -0.1, 0.1)
    embedding_matrix[vocab.word2idx[ModelConfig.PAD_TOKEN]] = torch.zeros(Word2VecConfig.VECTOR_SIZE)
    
    words_found = 0
    for word, idx in vocab.word2idx.items():
        if word in w2v_model.wv:
            # .copy() is needed to safely convert from NumPy array to PyTorch tensor
            embedding_matrix[idx] = torch.tensor(w2v_model.wv[word].copy())
            words_found += 1
            
    print(f"Loaded {words_found}/{vocab.vocab_size} vectors from custom Word2Vec.")
    return embedding_matrix, Word2VecConfig.VECTOR_SIZE

def prepare_data_for_dataset(dataset_name: str, preprocessor: TextPreprocessor):
    """Loads dataset, builds vocabulary, and returns DataLoaders."""
    print(f"\nLoading dataset: {dataset_name.upper()}...")
    
    if dataset_name == 'rotten_tomatoes':
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
        val_texts, val_labels = dataset['validation']['text'], dataset['validation']['label']
        test_texts, test_labels = dataset['test']['text'], dataset['test']['label']
        
    elif dataset_name == 'imdb':
        dataset = load_dataset("stanfordnlp/imdb")
        full_train = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_texts, train_labels = full_train['train']['text'], full_train['train']['label']
        val_texts, val_labels = full_train['test']['text'], full_train['test']['label']
        test_texts, test_labels = dataset['test']['text'], dataset['test']['label']
    
    print("Tokenizing training set to build vocabulary...")
    tokenized_train = [preprocessor.tokenize(text) for text in train_texts]
    vocab = Vocabulary()
    vocab.build_vocab(tokenized_train, min_count=2)
    print(f"Vocabulary size created: {vocab.vocab_size}")
    
    train_dataset = TextDataset(train_texts, train_labels, vocab, preprocessor)
    val_dataset = TextDataset(val_texts, val_labels, vocab, preprocessor)
    test_dataset = TextDataset(test_texts, test_labels, vocab, preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)
    
    return vocab, train_loader, val_loader, test_loader

def run_all_experiments(transformer_model_name: str = 'distilbert-base-uncased'):
    """Runs all 24 classical combinations + 2 Transformer experiments automatically."""
    datasets = ['rotten_tomatoes', 'imdb']
    models = ['CNN', 'LSTM']
    embeddings = ['Random', 'GloVe', 'Word2Vec']
    freezing_options = [True, False]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting fully automated run on device: {device}")
    
    preprocessor = TextPreprocessor()
    
    total_classical_exp = len(datasets) * len(models) * len(embeddings) * len(freezing_options)
    current_exp = 1

    # --- PART 1: CNN & LSTM EXPERIMENTS ---
    for dataset_name in datasets:
        vocab, train_loader, val_loader, test_loader = prepare_data_for_dataset(dataset_name, preprocessor)
        
        for emb_choice in embeddings:
            embedding_matrix = None
            embedding_dim = Word2VecConfig.VECTOR_SIZE
            
            if emb_choice == 'GloVe':
                embedding_matrix, embedding_dim = load_pretrained_glove(vocab)
            elif emb_choice == 'Word2Vec':
                embedding_matrix, embedding_dim = create_our_w2v_matrix(vocab)
            
            original_vector_size = Word2VecConfig.VECTOR_SIZE
            Word2VecConfig.VECTOR_SIZE = embedding_dim
            
            for model_type in models:
                for freeze_embedding in freezing_options:
                    experiment_name = f"{dataset_name}_{model_type}_{emb_choice}_Freeze{freeze_embedding}"
                    
                    print(f"\n{'='*50}")
                    print(f"Classical Experiment {current_exp}/{total_classical_exp}: {experiment_name}")
                    print(f"{'='*50}")
                    
                    if model_type == 'CNN':
                        model = TextCNN(vocab.vocab_size, embedding_matrix, freeze_embedding)
                    else:
                        model = TextLSTM(vocab.vocab_size, embedding_matrix, freeze_embedding)
                        
                    trainer = ModelTrainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        learning_rate=ModelConfig.LEARNING_RATE,
                        device=device,
                        experiment_name=experiment_name
                    )
                    
                    # 5 Epochs for CNN and LSTM
                    trainer.train(num_epochs=5)
                    current_exp += 1
            
            Word2VecConfig.VECTOR_SIZE = original_vector_size
            
    # --- PART 2: TRANSFORMER EXPERIMENTS ---
    print("\n" + "="*50)
    print(f" STARTING TRANSFORMER EXPERIMENTS WITH {transformer_model_name.upper()}")
    print("="*50)
    
    for dataset_name in datasets:
        # 3 epochs is standard for fine-tuning Transformers
        train_transformer(dataset_name=dataset_name, model_name=transformer_model_name, batch_size=16, num_epochs=3)

    print("\nAll 26 automated experiments (Classical + Transformers) finished successfully!")
    print("Check the 'results' folder for your CSV files and graphs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all deep learning experiments")
    parser.add_argument("--transformer_model", type=str, default="distilbert-base-uncased", help="HuggingFace model name for the transformer experiments")
    args = parser.parse_args()

    # Ensure torchtext is installed for GloVe
    try:
        import torchtext
    except ImportError:
        print("Error: 'torchtext' is required for pretrained GloVe embeddings.")
        print("Please install it using: pip install torchtext")
        exit(1)
        
    run_all_experiments(transformer_model_name=args.transformer_model)