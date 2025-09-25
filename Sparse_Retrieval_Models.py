import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod
import math
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class VietnameseDataset:
    """Dataset class for Vietnamese Q&A and document data."""
    
    def __init__(self, train_path: str, articles_path: str):
        self.train_path = Path(train_path)
        self.articles_path = Path(articles_path)
        self._validate_paths()
        
        (self.questions, self.true_keys, self.documents, 
         self.document_urls, self.key2docidx) = self._load_data()
    
    def _validate_paths(self) -> None:
        """Validate that input files exist."""
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_path}")
        if not self.articles_path.exists():
            raise FileNotFoundError(f"Articles file not found: {self.articles_path}")
    
    def _load_data(self) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, int]]:
        """Load and process data from JSON files."""
        try:
            df_questions = pd.read_json(self.train_path)
            df_documents = pd.read_json(self.articles_path)
            
            # Validate required columns
            self._validate_dataframe(df_questions, ["question", "article_url"], "questions")
            self._validate_dataframe(df_documents, ["document_text", "article_url"], "documents")
            
            questions = df_questions["question"].tolist()
            true_keys = df_questions["article_url"].tolist()
            documents = df_documents["document_text"].tolist()
            document_urls = df_documents["article_url"].tolist()
            
            key2docidx = {url: idx for idx, url in enumerate(document_urls)}
            
            return questions, true_keys, documents, document_urls, key2docidx
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
        """Validate DataFrame has required columns."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in {df_name} DataFrame: {missing_columns}")
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        return {
            "num_questions": len(self.questions),
            "num_documents": len(self.documents),
            "num_unique_urls": len(set(self.document_urls))
        }


class TextPreprocessor:
    """Text preprocessing utilities for Vietnamese text."""
    
    @staticmethod
    def preprocess_text(text: str, lowercase: bool = True, remove_punctuation: bool = True) -> str:
        """Basic text preprocessing."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if lowercase:
            text = text.lower()
        
        if remove_punctuation:
            # Remove punctuation but keep Vietnamese characters
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace."""
        return text.split()


class SparseRetrievalModel(ABC):
    """Abstract base class for sparse retrieval models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, documents: List[str]) -> None:
        """Fit the model on the document collection."""
        pass
    
    @abstractmethod
    def score(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Score documents for a query and return top-k results."""
        pass
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information."""
        return {"model_name": self.name}


class BM25Model(SparseRetrievalModel):
    """BM25 retrieval model implementation."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        super().__init__("BM25")
        self.k1 = k1
        self.b = b
        self.preprocessor = TextPreprocessor()
        
        # Model state
        self.documents = []
        self.processed_docs = []
        self.doc_lengths = []
        self.avgdl = 0.0
        self.doc_frequencies = Counter()
        self.idf_scores = {}
        self.N = 0
    
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 model on document collection."""
        print(f"Fitting {self.name} model on {len(documents)} documents...")
        
        self.documents = documents
        self.N = len(documents)
        
        # Preprocess documents
        print("Preprocessing documents...")
        self.processed_docs = []
        self.doc_lengths = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            processed_doc = self.preprocessor.preprocess_text(doc)
            tokens = self.preprocessor.tokenize(processed_doc)
            self.processed_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
        
        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate document frequencies
        print("Calculating document frequencies...")
        self.doc_frequencies = Counter()
        for tokens in tqdm(self.processed_docs, desc="Calculating DF"):
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_frequencies[token] += 1
        
        # Calculate IDF scores
        print("Calculating IDF scores...")
        self.idf_scores = {}
        for term, df in self.doc_frequencies.items():
            self.idf_scores[term] = math.log((self.N - df + 0.5) / (df + 0.5))
        
        self.is_fitted = True
        print(f"BM25 model fitted. Vocabulary size: {len(self.doc_frequencies)}")
    
    def score(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Score documents using BM25."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess_text(query)
        query_tokens = self.preprocessor.tokenize(processed_query)
        
        # Calculate BM25 scores for all documents
        scores = []
        
        for doc_idx, (doc_tokens, doc_len) in enumerate(zip(self.processed_docs, self.doc_lengths)):
            score = 0.0
            doc_token_counts = Counter(doc_tokens)
            
            for term in query_tokens:
                if term in self.idf_scores:
                    tf = doc_token_counts.get(term, 0)
                    idf = self.idf_scores[term]
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)
            
            scores.append((doc_idx, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def get_model_info(self) -> Dict[str, Union[str, float]]:
        """Get BM25 model information."""
        return {
            "model_name": self.name,
            "k1": self.k1,
            "b": self.b,
            "vocabulary_size": len(self.doc_frequencies),
            "avg_doc_length": self.avgdl
        }


class TFIDFModel(SparseRetrievalModel):
    """TF-IDF retrieval model implementation."""
    
    def __init__(self, max_features: Optional[int] = None, ngram_range: Tuple[int, int] = (1, 1)):
        super().__init__("TF-IDF")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.preprocessor = TextPreprocessor()
        
        # Model components
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = []
    
    def fit(self, documents: List[str]) -> None:
        """Fit TF-IDF model on document collection."""
        print(f"Fitting {self.name} model on {len(documents)} documents...")
        
        self.documents = documents
        
        # Preprocess documents
        print("Preprocessing documents...")
        processed_docs = []
        for doc in tqdm(documents, desc="Processing documents"):
            processed_doc = self.preprocessor.preprocess_text(doc)
            processed_docs.append(processed_doc)
        
        # Create and fit TF-IDF vectorizer
        print("Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(processed_docs)
        
        self.is_fitted = True
        print(f"TF-IDF model fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def score(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Score documents using TF-IDF cosine similarity."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Preprocess and vectorize query
        processed_query = self.preprocessor.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top-k documents
        top_indices = np.argsort(similarities)[::-1][:k]
        scores = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return scores
    
    def get_model_info(self) -> Dict[str, Union[str, int, Tuple]]:
        """Get TF-IDF model information."""
        vocab_size = len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        return {
            "model_name": self.name,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "vocabulary_size": vocab_size
        }


class QueryLikelihoodModel(SparseRetrievalModel):
    """Query Likelihood Model with Dirichlet smoothing."""
    
    def __init__(self, mu: float = 2000.0):
        super().__init__("Query Likelihood Model")
        self.mu = mu
        self.preprocessor = TextPreprocessor()
        
        # Model state
        self.documents = []
        self.processed_docs = []
        self.doc_lengths = []
        self.collection_model = Counter()
        self.total_collection_length = 0
    
    def fit(self, documents: List[str]) -> None:
        """Fit Query Likelihood model on document collection."""
        print(f"Fitting {self.name} model on {len(documents)} documents...")
        
        self.documents = documents
        
        # Preprocess documents
        print("Preprocessing documents...")
        self.processed_docs = []
        self.doc_lengths = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            processed_doc = self.preprocessor.preprocess_text(doc)
            tokens = self.preprocessor.tokenize(processed_doc)
            self.processed_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
        
        # Build collection model
        print("Building collection language model...")
        self.collection_model = Counter()
        for tokens in tqdm(self.processed_docs, desc="Building collection model"):
            self.collection_model.update(tokens)
        
        self.total_collection_length = sum(self.collection_model.values())
        
        self.is_fitted = True
        print(f"QLM model fitted. Vocabulary size: {len(self.collection_model)}")
    
    def score(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Score documents using Query Likelihood with Dirichlet smoothing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess_text(query)
        query_tokens = self.preprocessor.tokenize(processed_query)
        
        if not query_tokens:
            return []
        
        # Calculate query likelihood for all documents
        scores = []
        
        for doc_idx, (doc_tokens, doc_len) in enumerate(zip(self.processed_docs, self.doc_lengths)):
            doc_model = Counter(doc_tokens)
            log_likelihood = 0.0
            
            for term in query_tokens:
                # Dirichlet smoothing
                tf = doc_model.get(term, 0)
                cf = self.collection_model.get(term, 0)
                
                # P(term|doc) with Dirichlet smoothing
                p_term_doc = (tf + self.mu * (cf / self.total_collection_length)) / (doc_len + self.mu)
                
                # Avoid log(0)
                if p_term_doc > 0:
                    log_likelihood += math.log(p_term_doc)
                else:
                    log_likelihood += -float('inf')
                    break
            
            scores.append((doc_idx, log_likelihood))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def get_model_info(self) -> Dict[str, Union[str, float, int]]:
        """Get QLM model information."""
        return {
            "model_name": self.name,
            "mu": self.mu,
            "vocabulary_size": len(self.collection_model),
            "total_collection_length": self.total_collection_length
        }


class CSVLogger:
    """Class for logging evaluation data to CSV files."""
    
    def __init__(self, base_filename: str):
        self.base_filename = base_filename
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define CSV file paths
        self.metrics_csv = f"{base_filename}_metrics_{self.timestamp}.csv"
        self.detailed_csv = f"{base_filename}_detailed_{self.timestamp}.csv"
        self.search_logs_csv = f"{base_filename}_search_logs_{self.timestamp}.csv"
        
        # Initialize CSV files with headers
        self._initialize_csv_files()
    
    def _initialize_csv_files(self) -> None:
        """Initialize CSV files with appropriate headers."""
        # Metrics CSV
        with open(self.metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'model_name', 'model_params', 'num_questions', 
                'num_documents', 'valid_samples', 'total_samples', 'MRR',
                'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10'
            ])
        
        # Detailed results CSV
        with open(self.detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'question', 'true_document_url', 'true_document_index',
                'rank_of_true_doc', 'found_in_top_10', 'top_1_score', 'top_1_url',
                'top_2_score', 'top_2_url', 'top_3_score', 'top_3_url'
            ])
        
        # Search logs CSV
        with open(self.search_logs_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'question_length', 'search_timestamp', 'retrieved_rank',
                'document_index', 'similarity_score', 'document_url', 'is_relevant'
            ])
    
    def log_metrics(self, model_info: Dict, dataset_stats: Dict, metrics: Dict) -> None:
        """Log overall metrics to CSV."""
        with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                model_info.get('model_name', 'Unknown'),
                json.dumps(model_info),
                dataset_stats.get('num_questions', 0),
                dataset_stats.get('num_documents', 0),
                metrics.get('Valid_Samples', 0),
                metrics.get('Total_Samples', 0),
                metrics.get('MRR', 0),
                metrics.get('Recall@1', 0),
                metrics.get('Recall@3', 0),
                metrics.get('Recall@5', 0),
                metrics.get('Recall@10', 0)
            ])
    
    def log_detailed_results(self, sample_results: List[Dict]) -> None:
        """Log detailed results for each sample to CSV."""
        with open(self.detailed_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for sample in sample_results:
                # Find rank of true document
                rank_of_true = None
                for doc in sample['top_10_relevant_documents']:
                    if doc['is_relevant']:
                        rank_of_true = doc['rank']
                        break
                
                # Get top 3 documents info
                top_docs = sample['top_10_relevant_documents'][:3]
                
                row = [
                    sample['sample_id'],
                    sample['question'][:200],  # Truncate long questions
                    sample['true_document_url'],
                    sample['true_document_index'],
                    rank_of_true if rank_of_true else 'Not found',
                    rank_of_true is not None and rank_of_true <= 10,
                ]
                
                # Add top 3 document info
                for i in range(3):
                    if i < len(top_docs):
                        row.extend([top_docs[i]['similarity_score'], top_docs[i]['document_url']])
                    else:
                        row.extend(['', ''])
                
                writer.writerow(row)
    
    def log_search_results(self, sample_id: int, question: str, 
                          retrieved_docs: List[Dict]) -> None:
        """Log individual search results to CSV."""
        with open(self.search_logs_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            search_timestamp = datetime.now().isoformat()
            question_length = len(question.split())
            
            for doc in retrieved_docs:
                writer.writerow([
                    sample_id,
                    question_length,
                    search_timestamp,
                    doc['rank'],
                    doc['document_index'],
                    doc['similarity_score'],
                    doc['document_url'],
                    doc['is_relevant']
                ])
    
    def get_csv_paths(self) -> Dict[str, str]:
        """Return paths to all CSV files."""
        return {
            'metrics': self.metrics_csv,
            'detailed': self.detailed_csv,
            'search_logs': self.search_logs_csv
        }


class EvaluationMetrics:
    """Class for calculating retrieval metrics."""
    
    @staticmethod
    def calculate_metrics(model: SparseRetrievalModel,
                         questions: List[str],
                         true_keys: List[str],
                         document_urls: List[str],
                         key2docidx: Dict[str, int],
                         k_values: List[int] = [1, 3, 5, 10],
                         csv_logger: Optional[CSVLogger] = None) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Calculate retrieval metrics for sparse retrieval models.
        
        Returns:
            Tuple of (metrics_dict, sample_results_list)
        """
        max_k = max(k_values)
        n = len(questions)
        
        if n == 0:
            return {}, []
        
        # Initialize metrics
        metrics = {f"Recall@{k}": 0 for k in k_values}
        mrr_total = 0
        valid_samples = 0
        sample_results = []
        
        progress_bar = tqdm(
            zip(questions, true_keys),
            total=n,
            desc="Calculating metrics"
        )
        
        for j, (question, true_key) in enumerate(progress_bar):
            if true_key not in key2docidx:
                continue
            
            valid_samples += 1
            true_doc_idx = key2docidx[true_key]
            
            # Get retrieval results
            try:
                retrieval_results = model.score(question, k=max_k)
            except Exception as e:
                logging.warning(f"Error retrieving for question {j}: {str(e)}")
                continue
            
            # Build top 10 results
            top_10_docs = []
            for i, (doc_idx, score) in enumerate(retrieval_results[:10]):
                doc_url = document_urls[doc_idx] if doc_idx < len(document_urls) else "unknown"
                
                top_10_docs.append({
                    "rank": i + 1,
                    "document_index": int(doc_idx),
                    "document_url": doc_url,
                    "similarity_score": float(score),
                    "is_relevant": doc_idx == true_doc_idx
                })
            
            # Log search results to CSV if logger provided
            if csv_logger:
                csv_logger.log_search_results(j, question, top_10_docs)
            
            # Calculate recall metrics
            retrieved_doc_indices = [doc_idx for doc_idx, _ in retrieval_results]
            for k in k_values:
                if true_doc_idx in retrieved_doc_indices[:k]:
                    metrics[f"Recall@{k}"] += 1
            
            # Calculate MRR
            if true_doc_idx in retrieved_doc_indices:
                rank = retrieved_doc_indices.index(true_doc_idx) + 1
                mrr_total += 1.0 / rank
            
            sample_results.append({
                "sample_id": j,
                "question": question,
                "true_document_url": true_key,
                "true_document_index": int(true_doc_idx),
                "top_10_relevant_documents": top_10_docs
            })
        
        # Finalize metrics
        if valid_samples > 0:
            for k in k_values:
                metrics[f"Recall@{k}"] = metrics[f"Recall@{k}"] / valid_samples
            metrics["MRR"] = mrr_total / valid_samples
        
        metrics["Valid_Samples"] = valid_samples
        metrics["Total_Samples"] = n
        
        return metrics, sample_results


class SparseRetrievalEvaluator:
    """Main evaluator class for sparse retrieval models."""
    
    def __init__(self, 
                 train_path: str,
                 articles_path: str,
                 model: SparseRetrievalModel):
        self.dataset = VietnameseDataset(train_path, articles_path)
        self.model = model
        self.csv_logger: Optional[CSVLogger] = None
    
    def run_evaluation(self, 
                      k_values: List[int] = [1, 3, 5, 10],
                      output_file: str = "vietnamese_sparse_retrieval_evaluation_results.json",
                      enable_csv_logging: bool = True,
                      csv_base_filename: str = "vietnamese_sparse_retrieval_eval") -> Dict[str, Union[Dict, List]]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            k_values: K values for recall calculation
            output_file: Output file path for results
            enable_csv_logging: Whether to enable CSV logging
            csv_base_filename: Base filename for CSV logs
            
        Returns:
            Dictionary containing metrics and sample results
        """
        print("Starting sparse retrieval evaluation...")
        print(f"Dataset stats: {self.dataset.get_stats()}")
        print(f"Model: {self.model.name}")
        
        # Initialize CSV logger if enabled
        if enable_csv_logging:
            self.csv_logger = CSVLogger(csv_base_filename)
            print(f"CSV logging enabled. Files will be saved with timestamp.")
        
        # Fit model on documents
        print("\n1. Fitting retrieval model...")
        self.model.fit(self.dataset.documents)
        
        # Calculate metrics
        print("\n2. Calculating metrics...")
        metrics, sample_results = EvaluationMetrics.calculate_metrics(
            self.model,
            self.dataset.questions,
            self.dataset.true_keys,
            self.dataset.document_urls,
            self.dataset.key2docidx,
            k_values,
            self.csv_logger
        )
        
        # Log to CSV files
        if self.csv_logger:
            print("\n3. Saving CSV logs...")
            self.csv_logger.log_metrics(
                self.model.get_model_info(),
                self.dataset.get_stats(), 
                metrics
            )
            self.csv_logger.log_detailed_results(sample_results)
        
        # Save JSON results
        print(f"\n{'4' if self.csv_logger else '3'}. Saving JSON results...")
        output_path = self._save_results(
            self.model.get_model_info(),
            metrics,
            sample_results,
            output_file
        )
        
        # Display results
        csv_paths = self.csv_logger.get_csv_paths() if self.csv_logger else {}
        self._display_results(self.model.get_model_info(), metrics, sample_results, output_path, csv_paths)
        
        return {
            "metrics": metrics,
            "sample_results": sample_results,
            "output_file": output_path,
            "csv_files": csv_paths
        }
    
    def _save_results(self, 
                     model_info: Dict,
                     metrics: Dict[str, float],
                     sample_results: List[Dict],
                     output_file: str) -> str:
        """Save evaluation results to JSON file."""
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results = {
            "model_info": model_info,
            "evaluation_timestamp": datetime.now().isoformat(),
            "dataset_stats": self.dataset.get_stats(),
            "metrics": metrics,
            "sample_results": sample_results
        }
        
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
        
        return str(output_path)
    
    def _display_results(self, 
                        model_info: Dict,
                        metrics: Dict[str, float],
                        sample_results: List[Dict],
                        output_file: str,
                        csv_files: Dict[str, str] = {}) -> None:
        """Display evaluation results."""
        print("\n" + "="*60)
        print("SPARSE RETRIEVAL EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {model_info.get('model_name', 'Unknown')}")
        
        # Display model-specific parameters
        for key, value in model_info.items():
            if key != 'model_name':
                print(f"{key}: {value}")
        
        print(f"Valid samples: {metrics['Valid_Samples']}/{metrics['Total_Samples']}")
        
        print("\nMetrics:")
        print(f"  MRR: {metrics['MRR']:.4f}")
        for k in [1, 3, 5, 10]:
            if f"Recall@{k}" in metrics:
                print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
        
        print(f"\nJSON results saved to: {output_file}")
        
        if csv_files:
            print("\nCSV files saved:")
            print(f"  Metrics: {csv_files['metrics']}")
            print(f"  Detailed results: {csv_files['detailed']}")
            print(f"  Search logs: {csv_files['search_logs']}")
        
        print("\n" + "="*60)
        print("EXAMPLE RESULTS (First 3 samples)")
        print("="*60)
        
        for i, sample in enumerate(sample_results[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Question: {sample['question'][:100]}...")
            print(f"  True document: {sample['true_document_url']}")
            print("  Top 3 retrieved documents:")
            
            for doc in sample['top_10_relevant_documents'][:3]:
                status = "✓" if doc['is_relevant'] else "✗"
                print(f"    {doc['rank']}. [{status}] Score: {doc['similarity_score']:.4f}")
                print(f"       URL: {doc['document_url']}")


def create_model(model_type: str, **kwargs) -> SparseRetrievalModel:
    """Factory function to create sparse retrieval models."""
    model_type = model_type.lower()
    
    if model_type == "bm25":
        k1 = kwargs.get('k1', 1.2)
        b = kwargs.get('b', 0.75)
        return BM25Model(k1=k1, b=b)
    
    elif model_type == "tfidf":
        max_features = kwargs.get('max_features', None)
        ngram_range = kwargs.get('ngram_range', (1, 1))
        return TFIDFModel(max_features=max_features, ngram_range=ngram_range)
    
    elif model_type in ["qlm", "query_likelihood"]:
        mu = kwargs.get('mu', 2000.0)
        return QueryLikelihoodModel(mu=mu)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: bm25, tfidf, qlm")
