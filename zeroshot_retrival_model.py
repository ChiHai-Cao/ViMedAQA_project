import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging


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


class VietnameseEmbeddingModel:
    """Vietnamese document embedding model wrapper."""
    
    def __init__(self, 
                 model_name: str = "",
                 max_seq_length: int = 8192):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model: Optional[SentenceTransformer] = None
        self._is_loaded = False
    
    def load_model(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self.model.max_seq_length = self.max_seq_length
            self._is_loaded = True
            logging.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded or self.model is None:
            self.load_model()
    
    def encode_texts(self, 
                    texts: List[str], 
                    batch_size: int = 16,
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings with batch processing.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        self._ensure_model_loaded()
        
        if not texts:
            return np.array([])
        
        embeddings = []
        progress_bar = tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding texts",
            disable=not show_progress
        )
        
        try:
            for i in progress_bar:
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.append(batch_embeddings)
                
                progress_bar.set_postfix({
                    'processed': min(i + batch_size, len(texts)),
                    'total': len(texts)
                })
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {str(e)}")
        
        return np.vstack(embeddings) if embeddings else np.array([])


class FaissIndex:
    """FAISS index wrapper for similarity search."""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.index = self._build_index(embeddings)
        self.dimension = embeddings.shape[1]
    
    def _build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings."""
        if embeddings.size == 0:
            raise ValueError("Cannot build index from empty embeddings")
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings.astype(np.float32))
        return index
    
    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k similar documents.
        
        Args:
            query_embeddings: Query embeddings to search for
            k: Number of top results to return
            
        Returns:
            Tuple of (scores, indices)
        """
        if query_embeddings.size == 0:
            return np.array([]), np.array([])
        
        return self.index.search(query_embeddings.astype(np.float32), k)


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
                'timestamp', 'model_name', 'max_seq_length', 'num_questions', 
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
    
    def log_metrics(self, model_name: str, max_seq_length: int, 
                   dataset_stats: Dict, metrics: Dict) -> None:
        """Log overall metrics to CSV."""
        with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                model_name,
                max_seq_length,
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
    def calculate_metrics(index: FaissIndex,
                         q_embeddings: np.ndarray,
                         questions: List[str],
                         true_keys: List[str],
                         document_urls: List[str],
                         key2docidx: Dict[str, int],
                         k_values: List[int] = [1, 3, 5, 10],
                         csv_logger: Optional[CSVLogger] = None) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Calculate retrieval metrics.
        
        Returns:
            Tuple of (metrics_dict, sample_results_list)
        """
        max_k = max(k_values)
        n = len(q_embeddings)
        
        if n == 0:
            return {}, []
        
        scores, indices = index.search(q_embeddings, max_k)
        
        # Initialize metrics
        metrics = {f"Recall@{k}": 0 for k in k_values}
        mrr_total = 0
        valid_samples = 0
        sample_results = []
        
        progress_bar = tqdm(
            zip(indices, scores, questions, true_keys),
            total=n,
            desc="Calculating metrics"
        )
        
        for j, (question_indices, question_scores, question, true_key) in enumerate(progress_bar):
            if true_key not in key2docidx:
                continue
            
            valid_samples += 1
            true_doc_idx = key2docidx[true_key]
            
            # Build top 10 results
            top_10_docs = []
            for i in range(min(10, len(question_indices))):
                doc_idx = question_indices[i]
                score = float(question_scores[i])
                doc_url = document_urls[doc_idx] if doc_idx < len(document_urls) else "unknown"
                
                top_10_docs.append({
                    "rank": i + 1,
                    "document_index": int(doc_idx),
                    "document_url": doc_url,
                    "similarity_score": score,
                    "is_relevant": doc_idx == true_doc_idx
                })
            
            # Log search results to CSV if logger provided
            if csv_logger:
                csv_logger.log_search_results(j, question, top_10_docs)
            
            # Calculate recall metrics
            for k in k_values:
                if true_doc_idx in question_indices[:k]:
                    metrics[f"Recall@{k}"] += 1
            
            # Calculate MRR
            ranks = np.where(question_indices == true_doc_idx)[0]
            if len(ranks) > 0:
                mrr_total += 1.0 / (ranks[0] + 1)
            
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


class EmbeddingEvaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(self, 
                 train_path: str,
                 articles_path: str,
                 model_name: str = "dangvantuan/vietnamese-document-embedding",
                 max_seq_length: int = 8192):
        self.dataset = VietnameseDataset(train_path, articles_path)
        self.model = VietnameseEmbeddingModel(model_name, max_seq_length)
        self.index: Optional[FaissIndex] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.q_embeddings: Optional[np.ndarray] = None
        self.csv_logger: Optional[CSVLogger] = None
    
    def run_evaluation(self, 
                      batch_size: int = 16,
                      k_values: List[int] = [1, 3, 5, 10],
                      output_file: str = "vietnamese_embedding_evaluation_results.json",
                      enable_csv_logging: bool = True,
                      csv_base_filename: str = "vietnamese_embedding_eval") -> Dict[str, Union[Dict, List]]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            batch_size: Batch size for encoding
            k_values: K values for recall calculation
            output_file: Output file path for results
            enable_csv_logging: Whether to enable CSV logging
            csv_base_filename: Base filename for CSV logs
            
        Returns:
            Dictionary containing metrics and sample results
        """
        print("Starting evaluation...")
        print(f"Dataset stats: {self.dataset.get_stats()}")
        
        # Initialize CSV logger if enabled
        if enable_csv_logging:
            self.csv_logger = CSVLogger(csv_base_filename)
            print(f"CSV logging enabled. Files will be saved with timestamp.")
        
        # Load model
        print("\n1. Loading model...")
        self.model.load_model()
        
        # Encode documents
        print("\n2. Encoding documents...")
        self.doc_embeddings = self.model.encode_texts(
            self.dataset.documents, 
            batch_size=batch_size
        )
        
        # Build index
        print("\n3. Building FAISS index...")
        self.index = FaissIndex(self.doc_embeddings)
        
        # Encode questions
        print("\n4. Encoding questions...")
        self.q_embeddings = self.model.encode_texts(
            self.dataset.questions,
            batch_size=batch_size
        )
        
        # Calculate metrics
        print("\n5. Calculating metrics...")
        metrics, sample_results = EvaluationMetrics.calculate_metrics(
            self.index,
            self.q_embeddings,
            self.dataset.questions,
            self.dataset.true_keys,
            self.dataset.document_urls,
            self.dataset.key2docidx,
            k_values,
            self.csv_logger
        )
        
        # Log to CSV files
        if self.csv_logger:
            print("\n6. Saving CSV logs...")
            self.csv_logger.log_metrics(
                self.model.model_name, 
                self.model.max_seq_length, 
                self.dataset.get_stats(), 
                metrics
            )
            self.csv_logger.log_detailed_results(sample_results)
        
        # Save JSON results
        print(f"\n{'7' if self.csv_logger else '6'}. Saving JSON results...")
        output_path = self._save_results(
            self.model.model_name,
            metrics,
            sample_results,
            output_file
        )
        
        # Display results
        csv_paths = self.csv_logger.get_csv_paths() if self.csv_logger else {}
        self._display_results(self.model.model_name, metrics, sample_results, output_path, csv_paths)
        
        return {
            "metrics": metrics,
            "sample_results": sample_results,
            "output_file": output_path,
            "csv_files": csv_paths
        }
    
    def _save_results(self, 
                     model_name: str,
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
            "model_name": model_name,
            "max_seq_length": self.model.max_seq_length,
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
                        model_name: str,
                        metrics: Dict[str, float],
                        sample_results: List[Dict],
                        output_file: str,
                        csv_files: Dict[str, str] = {}) -> None:
        """Display evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Max sequence length: {self.model.max_seq_length:,} tokens")
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
