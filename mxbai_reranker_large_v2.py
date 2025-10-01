"""
Reranking Evaluation System (Standalone) - Enhanced Version
Author: AI Assistant
Date: 2025-10-01
This module provides a standalone pipeline for:
1. Cross-encoder reranking on pre-retrieved top-10 documents
2. Comprehensive evaluation and comparison
Assumes 'top10_retrieval.json' from retrieval phase exists.
Enhancements: More reranker configurability, error fixes, truncation handling.
"""
import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import torch
import warnings  # For truncation warnings

# Optional: For JSON schema validation
try:
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    warnings.warn("jsonschema not installed; skipping advanced JSON validation.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ============================================================================
# DATASET LOADER (REUSED FROM RETRIEVAL)
# ============================================================================
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
# ============================================================================
# RETRIEVAL RESULTS LOADER (ENHANCED WITH VALIDATION)
# ============================================================================
class RetrievalResultsLoader:
    """Load retrieval results from top-10 JSON file."""
   
    def __init__(self, top10_path: str):
        self.top10_path = Path(top10_path)
        self._validate_path()
        self.retrieval_results = self._load_results()
        self._validate_json_structure()  # New: Validate structure
   
    def _validate_path(self) -> None:
        """Validate that input file exists."""
        if not self.top10_path.exists():
            raise FileNotFoundError(f"Top-10 file not found: {self.top10_path}")
   
    def _load_results(self) -> List[Dict]:
        """Load retrieval results from JSON file."""
        try:
            with open(self.top10_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise RuntimeError(f"Error loading retrieval results: {str(e)}")
   
    def _validate_json_structure(self) -> None:
        """Validate JSON structure (basic schema check)."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "integer"},
                    "question": {"type": "string"},
                    "top_10_documents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rank": {"type": "integer"},
                                "document_index": {"type": "integer"},
                                "document_url": {"type": "string"},
                                "similarity_score": {"type": "number"}
                            },
                            "required": ["rank", "document_index", "document_url", "similarity_score"]
                        }
                    }
                },
                "required": ["question_id", "question", "top_10_documents"]
            }
        }
        
        if HAS_JSONSCHEMA:
            try:
                validate(instance=self.retrieval_results, schema=schema)
            except ValidationError as e:
                raise RuntimeError(f"JSON validation failed: {str(e)}")
        else:
            # Fallback: Basic key check
            for i, item in enumerate(self.retrieval_results):
                if not all(key in item for key in ["question_id", "question", "top_10_documents"]):
                    raise KeyError(f"Missing required keys in sample {i}: {list(item.keys())}")
                for j, doc in enumerate(item["top_10_documents"]):
                    if not all(key in doc for key in ["rank", "document_index", "document_url", "similarity_score"]):
                        raise KeyError(f"Missing doc keys in sample {i}, doc {j}")
   
    def __len__(self) -> int:
        return len(self.retrieval_results)
# ============================================================================
# RERANKER MODEL (ENHANCED CONFIGURABILITY)
# ============================================================================
class RerankerModel:
    """Cross-encoder reranker model wrapper with enhanced configurability."""
   
    def __init__(self, 
                 model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
                 max_length: int = 512,
                 device: Optional[str] = None,
                 trust_remote_code: bool = False,
                 default_batch_size: int = 32):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code
        self.default_batch_size = default_batch_size
        self.model: Optional[CrossEncoder] = None
        self._is_loaded = False
   
    def load_model(self) -> None:
        """Load the CrossEncoder model with current params."""
        try:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=self.trust_remote_code
            )
            self._is_loaded = True
            logging.info(f"Reranker model {self.model_name} loaded on {self.device} with max_length={self.max_length}")
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.warning("OOM detected; falling back to CPU and smaller batch.")
                self.device = "cpu"
                self.default_batch_size //= 2
                self.load_model()  # Retry
            else:
                raise RuntimeError(f"Failed to load reranker model: {str(e)}")
   
    def update_params(self, max_length: Optional[int] = None, device: Optional[str] = None, batch_size: Optional[int] = None):
        """Update model params runtime (reload if loaded)."""
        if max_length is not None:
            self.max_length = max_length
        if device is not None:
            self.device = device
        if batch_size is not None:
            self.default_batch_size = batch_size
        if self._is_loaded:
            self._is_loaded = False
            self.load_model()  # Reload with new params
   
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded or self.model is None:
            self.load_model()
   
    def truncate_docs(self, docs: List[str], warn_threshold: int = self.max_length) -> List[str]:
        """Truncate docs if longer than max_length and log warnings."""
        truncated = []
        for i, doc in enumerate(docs):
            if len(doc) > warn_threshold:
                logging.warning(f"Doc {i} truncated from {len(doc)} to {self.max_length} chars.")
                doc = doc[:self.max_length]
            truncated.append(doc)
        return truncated
   
    def rerank(self,
               query: str,
               documents: List[str],
               batch_size: Optional[int] = None) -> np.ndarray:
        """Rerank documents for a given query."""
        self._ensure_model_loaded()
        batch_size = batch_size or self.default_batch_size
       
        if not documents:
            return np.array([])
       
        # Truncate if needed
        documents = self.truncate_docs(documents)
       
        query_doc_pairs = [[query, doc] for doc in documents]
       
        scores = self.model.predict(
            query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )
       
        return np.array(scores)
# ============================================================================
# RERANKER CSV LOGGER (ENHANCED)
# ============================================================================
class RerankerCSVLogger:
    """Class for logging reranker evaluation data to CSV files."""
   
    def __init__(self, base_filename: str, truncate_question: int = 500):  # Configurable truncate
        self.base_filename = base_filename
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.truncate_question = truncate_question
       
        self.metrics_csv = f"{base_filename}_metrics_{self.timestamp}.csv"
        self.detailed_csv = f"{base_filename}_detailed_{self.timestamp}.csv"
        self.rerank_logs_csv = f"{base_filename}_rerank_logs_{self.timestamp}.csv"
        self.comparison_csv = f"{base_filename}_comparison_{self.timestamp}.csv"
       
        self._initialize_csv_files()
   
    def _initialize_csv_files(self) -> None:
        """Initialize CSV files with appropriate headers."""
        with open(self.metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'reranker_model', 'retriever_model', 'num_questions',
                'valid_samples', 'total_samples', 'MRR', 'Recall@1', 'Recall@3',
                'Recall@5', 'Recall@10', 'avg_rerank_time_ms'
            ])
       
        with open(self.detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'question', 'true_document_url', 'true_document_index',
                'rank_before_rerank', 'rank_after_rerank', 'improved', 'lost_relevant',  # New: lost_relevant
                'rerank_score', 'retrieval_score', 'top_1_url', 'top_1_rerank_score'
            ])
       
        with open(self.rerank_logs_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'question_length', 'rerank_timestamp', 'rank_after_rerank',
                'rank_before_rerank', 'document_index', 'rerank_score',
                'retrieval_score', 'document_url', 'is_relevant'
            ])
       
        with open(self.comparison_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'question', 'improved', 'rank_before', 'rank_after',
                'retrieval_score', 'rerank_score', 'rank_change'
            ])
   
    def log_metrics(self, reranker_model: str, retriever_model: str,
                   num_questions: int, metrics: Dict) -> None:
        """Log overall metrics to CSV."""
        with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                reranker_model,
                retriever_model,
                num_questions,
                metrics.get('Valid_Samples', 0),
                metrics.get('Total_Samples', 0),
                metrics.get('MRR', 0),
                metrics.get('Recall@1', 0),
                metrics.get('Recall@3', 0),
                metrics.get('Recall@5', 0),
                metrics.get('Recall@10', 0),
                metrics.get('avg_rerank_time_ms', 0)
            ])
   
    def log_detailed_results(self, sample_results: List[Dict]) -> None:
        """Log detailed results for each sample to CSV (no truncate or configurable)."""
        with open(self.detailed_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
           
            for sample in sample_results:
                rank_before = None
                rank_after = None
                rerank_score = None
                retrieval_score = None
                lost_relevant = True  # Default: assume lost if not found
               
                for doc in sample['reranked_documents']:
                    if doc['is_relevant']:
                        rank_after = doc['rank_after_rerank']
                        rank_before = doc['rank_before_rerank']
                        rerank_score = doc['rerank_score']
                        retrieval_score = doc['retrieval_score']
                        lost_relevant = False
                        break
               
                top_doc = sample['reranked_documents'][0] if sample['reranked_documents'] else {}
                improved = rank_before and rank_after and rank_after < rank_before
               
                q_text = sample['question'][:self.truncate_question] + "..." if len(sample['question']) > self.truncate_question else sample['question']
               
                writer.writerow([
                    sample['sample_id'],
                    q_text,
                    sample.get('true_document_url', ''),
                    sample.get('true_document_index', ''),
                    rank_before if rank_before else 'Not found',
                    rank_after if rank_after else 'Not found',
                    improved,
                    lost_relevant,  # New column
                    rerank_score if rerank_score else '',
                    retrieval_score if retrieval_score else '',
                    top_doc.get('document_url', ''),
                    top_doc.get('rerank_score', '')
                ])
   
    def log_rerank_results(self, sample_id: int, question: str,
                          reranked_docs: List[Dict]) -> None:
        """Log individual reranking results to CSV."""
        with open(self.rerank_logs_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
           
            rerank_timestamp = datetime.now().isoformat()
            question_length = len(question.split())
           
            for doc in reranked_docs:
                writer.writerow([
                    sample_id,
                    question_length,
                    rerank_timestamp,
                    doc['rank_after_rerank'],
                    doc['rank_before_rerank'],
                    doc['document_index'],
                    doc['rerank_score'],
                    doc['retrieval_score'],
                    doc['document_url'],
                    doc['is_relevant']
                ])
   
    def log_comparison(self, sample_results: List[Dict]) -> None:
        """Log before/after comparison results."""
        with open(self.comparison_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
           
            for sample in sample_results:
                rank_before = None
                rank_after = None
                rerank_score = None
                retrieval_score = None
               
                for doc in sample['reranked_documents']:
                    if doc['is_relevant']:
                        rank_after = doc['rank_after_rerank']
                        rank_before = doc['rank_before_rerank']
                        rerank_score = doc['rerank_score']
                        retrieval_score = doc['retrieval_score']
                        break
               
                if rank_before and rank_after:
                    improved = rank_after < rank_before
                    rank_change = rank_before - rank_after
                   
                    q_text = sample['question'][:self.truncate_question] + "..." if len(sample['question']) > self.truncate_question else sample['question']
                   
                    writer.writerow([
                        sample['sample_id'],
                        q_text,
                        improved,
                        rank_before,
                        rank_after,
                        retrieval_score,
                        rerank_score,
                        rank_change
                    ])
   
    def get_csv_paths(self) -> Dict[str, str]:
        """Return paths to all CSV files."""
        return {
            'metrics': self.metrics_csv,
            'detailed': self.detailed_csv,
            'rerank_logs': self.rerank_logs_csv,
            'comparison': self.comparison_csv
        }
# ============================================================================
# RERANKER METRICS (ENHANCED WITH TRUNCATION AND LOST LOGGING)
# ============================================================================
class RerankerMetrics:
    """Class for calculating reranking metrics."""
   
    @staticmethod
    def calculate_metrics(reranker: RerankerModel,
                         retrieval_results: List[Dict],
                         documents: List[str],
                         document_urls: List[str],
                         true_keys: List[str],
                         key2docidx: Dict[str, int],
                         k_values: List[int] = [1, 3, 5, 10],
                         batch_size: int = 32,
                         csv_logger: Optional[RerankerCSVLogger] = None) -> Tuple[Dict[str, float], List[Dict]]:
        """Calculate reranking metrics."""
        n = len(retrieval_results)
       
        if n == 0:
            return {}, []
       
        metrics = {f"Recall@{k}": 0 for k in k_values}
        mrr_total = 0
        valid_samples = 0
        sample_results = []
        total_rerank_time = 0
        lost_relevant_count = 0  # New: Track lost cases
       
        progress_bar = tqdm(
            enumerate(retrieval_results),
            total=n,
            desc="Reranking documents"
        )
       
        for j, result in progress_bar:
            question = result['question']
            question_id = result['question_id']
            retrieved_docs = result['top_10_documents']
           
            if question_id >= len(true_keys):
                continue
           
            true_key = true_keys[question_id]
            if true_key not in key2docidx:
                continue
           
            valid_samples += 1
            true_doc_idx = key2docidx[true_key]
           
            doc_indices = [doc['document_index'] for doc in retrieved_docs]
            doc_texts = [documents[idx] if idx < len(documents) else "Unknown document" for idx in doc_indices]  # Fallback text
            retrieval_scores = [doc['similarity_score'] for doc in retrieved_docs]
           
            start_time = datetime.now()
            rerank_scores = reranker.rerank(question, doc_texts, batch_size=batch_size)
            rerank_time = (datetime.now() - start_time).total_seconds() * 1000
            total_rerank_time += rerank_time
           
            sorted_indices = np.argsort(-rerank_scores)
           
            reranked_docs = []
            for new_rank, old_idx in enumerate(sorted_indices):
                doc_idx = doc_indices[old_idx]
                rerank_score = float(rerank_scores[old_idx])
                retrieval_score = retrieval_scores[old_idx]
                doc_url = document_urls[doc_idx] if doc_idx < len(document_urls) else "unknown"
                old_rank = retrieved_docs[old_idx]['rank']
               
                reranked_docs.append({
                    "rank_after_rerank": new_rank + 1,
                    "rank_before_rerank": old_rank,
                    "document_index": int(doc_idx),
                    "document_url": doc_url,
                    "rerank_score": rerank_score,
                    "retrieval_score": retrieval_score,
                    "is_relevant": doc_idx == true_doc_idx
                })
           
            if csv_logger:
                csv_logger.log_rerank_results(question_id, question, reranked_docs)
           
            reranked_doc_indices = [doc['document_index'] for doc in reranked_docs]
            for k in k_values:
                if true_doc_idx in reranked_doc_indices[:k]:
                    metrics[f"Recall@{k}"] += 1
           
            true_doc_new_rank = None
            try:
                true_doc_new_rank = next(
                    i for i, doc in enumerate(reranked_docs)
                    if doc['document_index'] == true_doc_idx
                )
                mrr_total += 1.0 / (true_doc_new_rank + 1)
            except StopIteration:
                lost_relevant_count += 1
                logging.warning(f"True doc {true_doc_idx} lost in rerank for sample {question_id}")
           
            sample_results.append({
                "sample_id": question_id,
                "question": question,
                "true_document_url": true_key,
                "true_document_index": int(true_doc_idx),
                "reranked_documents": reranked_docs,
                "rerank_time_ms": rerank_time,
                "lost_relevant": true_doc_new_rank is None  # New
            })
       
        if valid_samples > 0:
            for k in k_values:
                metrics[f"Recall@{k}"] = metrics[f"Recall@{k}"] / valid_samples
            metrics["MRR"] = mrr_total / valid_samples
            metrics["avg_rerank_time_ms"] = total_rerank_time / valid_samples
            metrics["lost_relevant_count"] = lost_relevant_count  # New metric
       
        metrics["Valid_Samples"] = valid_samples
        metrics["Total_Samples"] = n
       
        return metrics, sample_results
# ============================================================================
# RERANKER EVALUATOR (MAIN CLASS - ENHANCED)
# ============================================================================
class RerankerEvaluator:
    """Main evaluator class for reranking evaluation."""
   
    def __init__(self,
                 train_path: str,
                 articles_path: str,
                 top10_path: str,
                 reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v2",
                 retriever_model_name: str = "Unknown",
                 reranker_params: Optional[Dict] = None):  # New: Pass params
        self.dataset = VietnameseDataset(train_path, articles_path)
        self.retrieval_loader = RetrievalResultsLoader(top10_path)
        # Enhanced reranker init with params
        params = reranker_params or {}
        self.reranker = RerankerModel(
            model_name=reranker_model,
            max_length=params.get('max_length', 512),
            device=params.get('device'),
            trust_remote_code=params.get('trust_remote_code', False),
            default_batch_size=params.get('default_batch_size', 32)
        )
        self.retriever_model_name = retriever_model_name
        self.csv_logger: Optional[RerankerCSVLogger] = None
   
    def update_reranker_params(self, **kwargs):
        """Update reranker params runtime."""
        self.reranker.update_params(**kwargs)
   
    def run_evaluation(self,
                      batch_size: int = 32,
                      k_values: List[int] = [1, 3, 5, 10],
                      output_file: str = "reranker_results.json",
                      reranked_top10_file: str = "reranked_top10.json",
                      enable_csv_logging: bool = True,
                      csv_base_filename: str = "reranker_eval",
                      truncate_question: int = 500) -> Dict[str, Union[Dict, List]]:  # New param
        """Run the complete reranking evaluation pipeline."""
        print("Starting reranker evaluation...")
        print(f"Dataset stats: {self.dataset.get_stats()}")
        print(f"Questions to rerank: {len(self.retrieval_loader)}")
       
        if enable_csv_logging:
            self.csv_logger = RerankerCSVLogger(csv_base_filename, truncate_question=truncate_question)
       
        print("\n1. Loading reranker model...")
        self.reranker.load_model()
       
        print("\n2. Reranking documents...")
        metrics, sample_results = RerankerMetrics.calculate_metrics(
            self.reranker,
            self.retrieval_loader.retrieval_results,
            self.dataset.documents,
            self.dataset.document_urls,
            self.dataset.true_keys,
            self.dataset.key2docidx,
            k_values,
            batch_size,
            self.csv_logger
        )
       
        if self.csv_logger:
            print("\n3. Saving CSV logs...")
            self.csv_logger.log_metrics(
                self.reranker.model_name,
                self.retriever_model_name,
                len(self.retrieval_loader),
                metrics
            )
            self.csv_logger.log_detailed_results(sample_results)
            self.csv_logger.log_comparison(sample_results)
       
        print(f"\n{'4' if self.csv_logger else '3'}. Saving reranked top-10...")
        reranked_path = self._save_reranked_top10(sample_results, reranked_top10_file)
       
        print(f"\n{'5' if self.csv_logger else '4'}. Saving results...")
        output_path = self._save_results(
            self.reranker.model_name,
            self.retriever_model_name,
            metrics,
            sample_results,
            output_file
        )
       
        csv_paths = self.csv_logger.get_csv_paths() if self.csv_logger else {}
        self._display_results(
            self.reranker.model_name,
            metrics,
            sample_results,
            output_path,
            reranked_path,
            csv_paths
        )
       
        return {
            "metrics": metrics,
            "sample_results": sample_results,
            "output_file": output_path,
            "reranked_top10_file": reranked_path,
            "csv_files": csv_paths
        }
   
    # _save_reranked_top10, _save_results, _display_results remain the same as original
    # (Omit for brevity; copy from original code)
    def _save_reranked_top10(self, sample_results: List[Dict], output_file: str) -> str:
        def convert_numpy_types(obj):
            if isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
       
        reranked_data = []
        for sample in sample_results:
            reranked_data.append({
                "question_id": sample["sample_id"],
                "question": sample["question"],
                "reranked_top_10_documents": sample["reranked_documents"]
            })
       
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reranked_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
       
        return str(output_path)
   
    def _save_results(self,
                     reranker_model: str,
                     retriever_model: str,
                     metrics: Dict[str, float],
                     sample_results: List[Dict],
                     output_file: str) -> str:
        def convert_numpy_types(obj):
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
            "reranker_model": reranker_model,
            "retriever_model": retriever_model,
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
                        reranker_model: str,
                        metrics: Dict[str, float],
                        sample_results: List[Dict],
                        output_file: str,
                        reranked_file: str,
                        csv_files: Dict[str, str] = {}) -> None:
        print("\n" + "="*60)
        print("RERANKER EVALUATION RESULTS")
        print("="*60)
        print(f"Reranker Model: {reranker_model}")
        print(f"Valid samples: {metrics['Valid_Samples']}/{metrics['Total_Samples']}")
        print(f"Avg rerank time: {metrics.get('avg_rerank_time_ms', 0):.2f} ms/question")
        if 'lost_relevant_count' in metrics:
            print(f"Lost relevant docs: {metrics['lost_relevant_count']}")
       
        print("\nMetrics (After Reranking):")
        print(f" MRR: {metrics['MRR']:.4f}")
        for k in [1, 3, 5, 10]:
            if f"Recall@{k}" in metrics:
                print(f" Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
       
        improved_count = 0
        unchanged_count = 0
        degraded_count = 0
       
        for sample in sample_results:
            rank_before = None
            rank_after = None
           
            for doc in sample['reranked_documents']:
                if doc['is_relevant']:
                    rank_before = doc['rank_before_rerank']
                    rank_after = doc['rank_after_rerank']
                    break
           
            if rank_before and rank_after:
                if rank_after < rank_before:
                    improved_count += 1
                elif rank_after == rank_before:
                    unchanged_count += 1
                else:
                    degraded_count += 1
       
        total = improved_count + unchanged_count + degraded_count
        if total > 0:
            print("\nReranking Impact:")
            print(f" Improved: {improved_count} ({improved_count/total*100:.1f}%)")
            print(f" Unchanged: {unchanged_count} ({unchanged_count/total*100:.1f}%)")
            print(f" Degraded: {degraded_count} ({degraded_count/total*100:.1f}%)")
       
        print(f"\nResults saved to: {output_file}")
        print(f"Reranked top-10 saved to: {reranked_file}")
       
        if csv_files:
            print("\nCSV files:")
            for name, path in csv_files.items():
                print(f" {name}: {path}")
