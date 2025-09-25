import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from collections import Counter
import re
import math


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


class VietnameseTextProcessor:
    """Vietnamese text preprocessing for BM25."""
    
    def __init__(self):
        # Vietnamese stop words (basic set)
        self.stop_words = {
            'là', 'của', 'và', 'có', 'được', 'trong', 'với', 'để', 'cho', 'từ', 'trên',
            'về', 'tại', 'một', 'các', 'những', 'này', 'đó', 'hay', 'hoặc', 'như', 'sẽ',
            'đã', 'đang', 'phải', 'nên', 'cần', 'ra', 'vào', 'bị', 'bằng', 'theo', 'sau',
            'trước', 'giữa', 'dưới', 'lên', 'xuống', 'ngoài', 'cùng', 'nhau', 'thì', 'mà',
            'nếu', 'khi', 'lúc', 'bao', 'ai', 'gì', 'đâu', 'sao', 'nào', 'làm', 'việc',
            'người', 'ngày', 'năm', 'thời', 'lại', 'chỉ', 'rất', 'nhiều', 'ít', 'hơn',
            'nhất', 'cũng', 'còn', 'đều', 'luôn', 'vẫn', 'đã', 'sẽ', 'không', 'chưa',
            'bây', 'giờ', 'hiện', 'nay'
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess Vietnamese text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and keep only letters, numbers, and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove stop words and short tokens
        tokens = [token for token in tokens if len(token) > 1 and token not in self.stop_words]
        
        return tokens


class BM25Index:
    """BM25 index for text retrieval."""
    
    def __init__(self, documents: List[str], k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            documents: List of documents to index
            k1: BM25 parameter k1 (term frequency saturation)
            b: BM25 parameter b (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.processor = VietnameseTextProcessor()
        
        # Preprocess documents
        self.documents = documents
        self.tokenized_docs = [self.processor.preprocess_text(doc) for doc in documents]
        
        # Calculate statistics
        self.num_docs = len(self.tokenized_docs)
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0
        
        # Build vocabulary and document frequencies
        self.vocabulary = set()
        self.doc_freqs = {}
        
        for doc_tokens in self.tokenized_docs:
            unique_tokens = set(doc_tokens)
            self.vocabulary.update(unique_tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        # Pre-calculate IDF scores
        self.idf_scores = {}
        for term in self.vocabulary:
            df = self.doc_freqs[term]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5))
            self.idf_scores[term] = idf
        
        logging.info(f"BM25 index built: {self.num_docs} docs, {len(self.vocabulary)} terms")
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], doc_length: int) -> float:
        """Calculate BM25 score for a query-document pair."""
        score = 0.0
        term_counts = Counter(doc_tokens)
        
        for term in query_tokens:
            if term not in self.vocabulary:
                continue
            
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            
            idf = self.idf_scores[term]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, k: int = 10) -> Tuple[List[float], List[int]]:
        """
        Search for top-k documents using BM25.
        
        Args:
            query: Search query
            k: Number of top results to return
            
        Returns:
            Tuple of (scores, document_indices)
        """
        query_tokens = self.processor.preprocess_text(query)
        
        if not query_tokens:
            return [], []
        
        # Calculate scores for all documents
        scores = []
        for i, (doc_tokens, doc_length) in enumerate(zip(self.tokenized_docs, self.doc_lengths)):
            score = self._calculate_bm25_score(query_tokens, doc_tokens, doc_length)
            scores.append((score, i))
        
        # Sort by score (descending) and return top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k_scores = [score for score, _ in scores[:k]]
        top_k_indices = [idx for _, idx in scores[:k]]
        
        return top_k_scores, top_k_indices
    
    def batch_search(self, queries: List[str], k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of top results per query
            
        Returns:
            Tuple of (scores_matrix, indices_matrix)
        """
        all_scores = []
        all_indices = []
        
        for query in tqdm(queries, desc="Processing queries"):
            scores, indices = self.search(query, k)
            
            # Pad with zeros/invalid indices if needed
            while len(scores) < k:
                scores.append(0.0)
                indices.append(-1)
            
            all_scores.append(scores)
            all_indices.append(indices)
        
        return np.array(all_scores), np.array(all_indices)


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
                'timestamp', 'model_name', 'k1_param', 'b_param', 'num_questions', 
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
    
    def log_metrics(self, model_name: str, k1: float, b: float,
                   dataset_stats: Dict, metrics: Dict) -> None:
        """Log overall metrics to CSV."""
        with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                model_name,
                k1, b,
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
    def calculate_metrics(index: BM25Index,
                         questions: List[str],
                         true_keys: List[str],
                         document_urls: List[str],
                         key2docidx: Dict[str, int],
                         k_values: List[int] = [1, 3, 5, 10],
                         csv_logger: Optional[CSVLogger] = None) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Calculate retrieval metrics using BM25.
        
        Returns:
            Tuple of (metrics_dict, sample_results_list)
        """
        max_k = max(k_values)
        n = len(questions)
        
        if n == 0:
            return {}, []
        
        # Get search results for all queries
        scores, indices = index.batch_search(questions, max_k)
        
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
                if doc_idx == -1:  # Invalid index from padding
                    continue
                    
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
                valid_indices = [idx for idx in question_indices[:k] if idx != -1]
                if true_doc_idx in valid_indices:
                    metrics[f"Recall@{k}"] += 1
            
            # Calculate MRR
            valid_indices = [idx for idx in question_indices if idx != -1]
            ranks = [i for i, idx in enumerate(valid_indices) if idx == true_doc_idx]
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


class BM25Evaluator:
    """Main evaluator class that orchestrates the BM25 evaluation process."""
    
    def __init__(self, 
                 train_path: str,
                 articles_path: str,
                 k1: float = 1.2,
                 b: float = 0.75):
        """
        Initialize BM25 evaluator.
        
        Args:
            train_path: Path to training questions JSON
            articles_path: Path to articles JSON
            k1: BM25 parameter k1 (term frequency saturation)
            b: BM25 parameter b (length normalization)
        """
        self.dataset = VietnameseDataset(train_path, articles_path)
        self.k1 = k1
        self.b = b
        self.index: Optional[BM25Index] = None
        self.csv_logger: Optional[CSVLogger] = None
    
    def run_evaluation(self, 
                      k_values: List[int] = [1, 3, 5, 10],
                      output_file: str = "vietnamese_bm25_evaluation_results.json",
                      enable_csv_logging: bool = True,
                      csv_base_filename: str = "vietnamese_bm25_eval") -> Dict[str, Union[Dict, List]]:
        """
        Run the complete BM25 evaluation pipeline.
        
        Args:
            k_values: K values for recall calculation
            output_file: Output file path for results
            enable_csv_logging: Whether to enable CSV logging
            csv_base_filename: Base filename for CSV logs
            
        Returns:
            Dictionary containing metrics and sample results
        """
        print("Starting BM25 evaluation...")
        print(f"Dataset stats: {self.dataset.get_stats()}")
        print(f"BM25 parameters: k1={self.k1}, b={self.b}")
        
        # Initialize CSV logger if enabled
        if enable_csv_logging:
            self.csv_logger = CSVLogger(csv_base_filename)
            print(f"CSV logging enabled. Files will be saved with timestamp.")
        
        # Build BM25 index
        print("\n1. Building BM25 index...")
        self.index = BM25Index(self.dataset.documents, k1=self.k1, b=self.b)
        
        # Calculate metrics
        print("\n2. Calculating metrics...")
        metrics, sample_results = EvaluationMetrics.calculate_metrics(
            self.index,
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
                "BM25", 
                self.k1, 
                self.b,
                self.dataset.get_stats(), 
                metrics
            )
            self.csv_logger.log_detailed_results(sample_results)
        
        # Save JSON results
        print(f"\n{'4' if self.csv_logger else '3'}. Saving JSON results...")
        output_path = self._save_results(
            metrics,
            sample_results,
            output_file
        )
        
        # Display results
        csv_paths = self.csv_logger.get_csv_paths() if self.csv_logger else {}
        self._display_results(metrics, sample_results, output_path, csv_paths)
        
        return {
            "metrics": metrics,
            "sample_results": sample_results,
            "output_file": output_path,
            "csv_files": csv_paths
        }
    
    def _save_results(self, 
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
            "model_name": "BM25",
            "k1_parameter": self.k1,
            "b_parameter": self.b,
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
                        metrics: Dict[str, float],
                        sample_results: List[Dict],
                        output_file: str,
                        csv_files: Dict[str, str] = {}) -> None:
        """Display evaluation results."""
        print("\n" + "="*60)
        print("BM25 EVALUATION RESULTS")
        print("="*60)
        print(f"Model: BM25")
        print(f"Parameters: k1={self.k1}, b={self.b}")
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
