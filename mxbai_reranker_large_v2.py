"""
Reranking Evaluation System - Kaggle-Optimized Version
Fixes common errors and optimizes for Kaggle environment
"""
import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from tqdm.auto import tqdm
import torch
import warnings
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Optional: JSON schema validation
try:
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    logger.warning("jsonschema not installed; skipping validation.")

# Import sentence_transformers with error handling
try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
    raise

# ============================================================================
# DATASET LOADER
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
            # Read with error handling
            with open(self.train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            with open(self.articles_path, 'r', encoding='utf-8') as f:
                articles_data = json.load(f)
            
            df_questions = pd.DataFrame(train_data)
            df_documents = pd.DataFrame(articles_data)
           
            # Validate required columns
            self._validate_dataframe(df_questions, ["question", "article_url"], "questions")
            self._validate_dataframe(df_documents, ["document_text", "article_url"], "documents")
           
            questions = df_questions["question"].astype(str).tolist()
            true_keys = df_questions["article_url"].astype(str).tolist()
            documents = df_documents["document_text"].astype(str).tolist()
            document_urls = df_documents["article_url"].astype(str).tolist()
           
            key2docidx = {url: idx for idx, url in enumerate(document_urls)}
           
            return questions, true_keys, documents, document_urls, key2docidx
           
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
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
# RETRIEVAL RESULTS LOADER
# ============================================================================
class RetrievalResultsLoader:
    """Load retrieval results from top-10 JSON file."""
   
    def __init__(self, top10_path: str):
        self.top10_path = Path(top10_path)
        self._validate_path()
        self.retrieval_results = self._load_results()
        self._validate_json_structure()
   
    def _validate_path(self) -> None:
        """Validate that input file exists."""
        if not self.top10_path.exists():
            raise FileNotFoundError(f"Top-10 file not found: {self.top10_path}")
   
    def _load_results(self) -> List[Dict]:
        """Load retrieval results from JSON file."""
        try:
            with open(self.top10_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} retrieval results")
            return data
        except Exception as e:
            logger.error(f"Error loading retrieval results: {str(e)}")
            raise RuntimeError(f"Error loading retrieval results: {str(e)}")
   
    def _validate_json_structure(self) -> None:
        """Validate JSON structure (basic key check)."""
        required_keys = ["question_id", "question", "top_10_documents"]
        doc_keys = ["rank", "document_index", "document_url", "similarity_score"]
        
        for i, item in enumerate(self.retrieval_results):
            # Check main keys
            missing = [k for k in required_keys if k not in item]
            if missing:
                raise KeyError(f"Missing keys in sample {i}: {missing}")
            
            # Check document keys
            for j, doc in enumerate(item.get("top_10_documents", [])):
                missing_doc = [k for k in doc_keys if k not in doc]
                if missing_doc:
                    raise KeyError(f"Missing doc keys in sample {i}, doc {j}: {missing_doc}")
   
    def __len__(self) -> int:
        return len(self.retrieval_results)

# ============================================================================
# RERANKER MODEL
# ============================================================================
class RerankerModel:
    """Cross-encoder reranker model wrapper with Kaggle optimizations."""
   
    def __init__(self, 
                 model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
                 max_length: int = 512,
                 device: Optional[str] = None,
                 trust_remote_code: bool = False,
                 default_batch_size: int = 16):  # Reduced default for Kaggle
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or self._get_optimal_device()
        self.trust_remote_code = trust_remote_code
        self.default_batch_size = default_batch_size
        self.model: Optional[CrossEncoder] = None
        self._is_loaded = False
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for Kaggle."""
        if torch.cuda.is_available():
            # Check CUDA memory
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"CUDA available with {total_memory:.2f} GB memory")
                return "cuda"
            except:
                pass
        logger.info("Using CPU")
        return "cpu"
   
    def load_model(self) -> None:
        """Load the CrossEncoder model with error handling."""
        try:
            logger.info(f"Loading reranker: {self.model_name}")
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=self.trust_remote_code
            )
            self._is_loaded = True
            logger.info(f"Model loaded on {self.device} (max_length={self.max_length})")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                logger.warning("OOM detected; falling back to CPU")
                torch.cuda.empty_cache()
                gc.collect()
                self.device = "cpu"
                self.default_batch_size = max(4, self.default_batch_size // 4)
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code
                )
                self._is_loaded = True
            else:
                raise RuntimeError(f"Failed to load reranker model: {str(e)}")
   
    def update_params(self, max_length: Optional[int] = None, 
                     device: Optional[str] = None, 
                     batch_size: Optional[int] = None):
        """Update model params at runtime."""
        reload_needed = False
        if max_length is not None and max_length != self.max_length:
            self.max_length = max_length
            reload_needed = True
        if device is not None and device != self.device:
            self.device = device
            reload_needed = True
        if batch_size is not None:
            self.default_batch_size = batch_size
        
        if reload_needed and self._is_loaded:
            logger.info("Reloading model with new params")
            self._is_loaded = False
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            self.load_model()
   
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded or self.model is None:
            self.load_model()
   
    def rerank(self,
               query: str,
               documents: List[str],
               batch_size: Optional[int] = None) -> np.ndarray:
        """Rerank documents for a given query with error handling."""
        self._ensure_model_loaded()
        batch_size = batch_size or self.default_batch_size
       
        if not documents:
            return np.array([])
       
        # Prepare pairs
        query_doc_pairs = [[str(query), str(doc)[:self.max_length]] for doc in documents]
       
        try:
            scores = self.model.predict(
                query_doc_pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return np.array(scores)
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM during reranking, reducing batch size to {batch_size//2}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                return self.rerank(query, documents, batch_size=max(1, batch_size//2))
            else:
                raise

# ============================================================================
# CSV LOGGER
# ============================================================================
class RerankerCSVLogger:
    """CSV logger with Kaggle-friendly paths."""
   
    def __init__(self, base_filename: str, truncate_question: int = 200):
        self.base_filename = base_filename
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.truncate_question = truncate_question
       
        # Use simple filenames for Kaggle
        self.metrics_csv = f"{base_filename}_metrics.csv"
        self.detailed_csv = f"{base_filename}_detailed.csv"
        self.rerank_logs_csv = f"{base_filename}_rerank_logs.csv"
        self.comparison_csv = f"{base_filename}_comparison.csv"
       
        self._initialize_csv_files()
   
    def _initialize_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        # Metrics CSV
        with open(self.metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'reranker_model', 'retriever_model', 'num_questions',
                'valid_samples', 'total_samples', 'MRR', 'Recall@1', 'Recall@3',
                'Recall@5', 'Recall@10', 'avg_rerank_time_ms', 'lost_relevant_count'
            ])
       
        # Detailed CSV
        with open(self.detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'question', 'true_document_url', 'true_document_index',
                'rank_before_rerank', 'rank_after_rerank', 'improved', 'lost_relevant',
                'rerank_score', 'retrieval_score', 'top_1_url', 'top_1_rerank_score'
            ])
       
        # Rerank logs CSV
        with open(self.rerank_logs_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'question_length', 'rerank_timestamp', 'rank_after_rerank',
                'rank_before_rerank', 'document_index', 'rerank_score',
                'retrieval_score', 'document_url', 'is_relevant'
            ])
       
        # Comparison CSV
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
                metrics.get('avg_rerank_time_ms', 0),
                metrics.get('lost_relevant_count', 0)
            ])
   
    def log_detailed_results(self, sample_results: List[Dict]) -> None:
        """Log detailed results for each sample."""
        with open(self.detailed_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
           
            for sample in sample_results:
                rank_before = None
                rank_after = None
                rerank_score = None
                retrieval_score = None
                lost_relevant = True
               
                for doc in sample.get('reranked_documents', []):
                    if doc.get('is_relevant', False):
                        rank_after = doc['rank_after_rerank']
                        rank_before = doc['rank_before_rerank']
                        rerank_score = doc['rerank_score']
                        retrieval_score = doc['retrieval_score']
                        lost_relevant = False
                        break
               
                top_doc = sample.get('reranked_documents', [{}])[0]
                improved = (rank_before is not None and rank_after is not None 
                           and rank_after < rank_before)
               
                q_text = sample['question']
                if len(q_text) > self.truncate_question:
                    q_text = q_text[:self.truncate_question] + "..."
               
                writer.writerow([
                    sample['sample_id'],
                    q_text,
                    sample.get('true_document_url', ''),
                    sample.get('true_document_index', ''),
                    rank_before if rank_before else 'Not found',
                    rank_after if rank_after else 'Not found',
                    improved,
                    lost_relevant,
                    rerank_score if rerank_score else '',
                    retrieval_score if retrieval_score else '',
                    top_doc.get('document_url', ''),
                    top_doc.get('rerank_score', '')
                ])
   
    def log_rerank_results(self, sample_id: int, question: str,
                          reranked_docs: List[Dict]) -> None:
        """Log individual reranking results."""
        with open(self.rerank_logs_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            rerank_timestamp = datetime.now().isoformat()
            question_length = len(str(question).split())
           
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
        """Log before/after comparison."""
        with open(self.comparison_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
           
            for sample in sample_results:
                rank_before = None
                rank_after = None
                rerank_score = None
                retrieval_score = None
               
                for doc in sample.get('reranked_documents', []):
                    if doc.get('is_relevant', False):
                        rank_after = doc['rank_after_rerank']
                        rank_before = doc['rank_before_rerank']
                        rerank_score = doc['rerank_score']
                        retrieval_score = doc['retrieval_score']
                        break
               
                if rank_before and rank_after:
                    improved = rank_after < rank_before
                    rank_change = rank_before - rank_after
                   
                    q_text = sample['question']
                    if len(q_text) > self.truncate_question:
                        q_text = q_text[:self.truncate_question] + "..."
                   
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
# METRICS CALCULATOR
# ============================================================================
class RerankerMetrics:
    """Calculate reranking metrics with optimizations."""
   
    @staticmethod
    def calculate_metrics(reranker: RerankerModel,
                         retrieval_results: List[Dict],
                         documents: List[str],
                         document_urls: List[str],
                         true_keys: List[str],
                         key2docidx: Dict[str, int],
                         k_values: List[int] = [1, 3, 5, 10],
                         batch_size: int = 16,
                         csv_logger: Optional[RerankerCSVLogger] = None) -> Tuple[Dict[str, float], List[Dict]]:
        """Calculate reranking metrics."""
        n = len(retrieval_results)
        if n == 0:
            return {}, []
       
        metrics = {f"Recall@{k}": 0.0 for k in k_values}
        mrr_total = 0.0
        valid_samples = 0
        sample_results = []
        total_rerank_time = 0.0
        lost_relevant_count = 0
       
        progress_bar = tqdm(
            enumerate(retrieval_results),
            total=n,
            desc="Reranking",
            leave=True
        )
       
        for j, result in progress_bar:
            try:
                question = str(result['question'])
                question_id = int(result['question_id'])
                retrieved_docs = result['top_10_documents']
               
                if question_id >= len(true_keys):
                    logger.warning(f"Question ID {question_id} out of range")
                    continue
               
                true_key = true_keys[question_id]
                if true_key not in key2docidx:
                    logger.warning(f"True key {true_key} not in index")
                    continue
               
                valid_samples += 1
                true_doc_idx = key2docidx[true_key]
               
                # Get documents and scores
                doc_indices = [int(doc['document_index']) for doc in retrieved_docs]
                doc_texts = [
                    documents[idx] if idx < len(documents) else "Unknown document" 
                    for idx in doc_indices
                ]
                retrieval_scores = [float(doc['similarity_score']) for doc in retrieved_docs]
               
                # Rerank
                start_time = datetime.now()
                rerank_scores = reranker.rerank(question, doc_texts, batch_size=batch_size)
                rerank_time = (datetime.now() - start_time).total_seconds() * 1000
                total_rerank_time += rerank_time
               
                # Sort by rerank scores
                sorted_indices = np.argsort(-rerank_scores)
               
                # Build reranked documents
                reranked_docs = []
                for new_rank, old_idx in enumerate(sorted_indices):
                    doc_idx = doc_indices[old_idx]
                    rerank_score = float(rerank_scores[old_idx])
                    retrieval_score = retrieval_scores[old_idx]
                    doc_url = (document_urls[doc_idx] if doc_idx < len(document_urls) 
                              else "unknown")
                    old_rank = int(retrieved_docs[old_idx]['rank'])
                   
                    reranked_docs.append({
                        "rank_after_rerank": int(new_rank + 1),
                        "rank_before_rerank": int(old_rank),
                        "document_index": int(doc_idx),
                        "document_url": doc_url,
                        "rerank_score": float(rerank_score),
                        "retrieval_score": float(retrieval_score),
                        "is_relevant": bool(doc_idx == true_doc_idx)
                    })
               
                # Log to CSV
                if csv_logger:
                    csv_logger.log_rerank_results(question_id, question, reranked_docs)
               
                # Calculate metrics
                reranked_doc_indices = [doc['document_index'] for doc in reranked_docs]
                for k in k_values:
                    if true_doc_idx in reranked_doc_indices[:k]:
                        metrics[f"Recall@{k}"] += 1
               
                # Calculate MRR
                true_doc_new_rank = None
                for i, doc in enumerate(reranked_docs):
                    if doc['document_index'] == true_doc_idx:
                        true_doc_new_rank = i
                        mrr_total += 1.0 / (i + 1)
                        break
                
                if true_doc_new_rank is None:
                    lost_relevant_count += 1
               
                sample_results.append({
                    "sample_id": int(question_id),
                    "question": question,
                    "true_document_url": true_key,
                    "true_document_index": int(true_doc_idx),
                    "reranked_documents": reranked_docs,
                    "rerank_time_ms": float(rerank_time),
                    "lost_relevant": bool(true_doc_new_rank is None)
                })
                
                # Update progress bar
                if (j + 1) % 10 == 0:
                    progress_bar.set_postfix({
                        'valid': valid_samples,
                        'MRR': f"{mrr_total/valid_samples:.3f}" if valid_samples > 0 else "0"
                    })
            
            except Exception as e:
                logger.error(f"Error processing sample {j}: {str(e)}")
                continue
       
        # Calculate final metrics
        if valid_samples > 0:
            for k in k_values:
                metrics[f"Recall@{k}"] = float(metrics[f"Recall@{k}"] / valid_samples)
            metrics["MRR"] = float(mrr_total / valid_samples)
            metrics["avg_rerank_time_ms"] = float(total_rerank_time / valid_samples)
            metrics["lost_relevant_count"] = int(lost_relevant_count)
       
        metrics["Valid_Samples"] = int(valid_samples)
        metrics["Total_Samples"] = int(n)
       
        return metrics, sample_results

# ============================================================================
# MAIN EVALUATOR
# ============================================================================
class RerankerEvaluator:
    """Main evaluator class for reranking."""
   
    def __init__(self,
                 train_path: str,
                 articles_path: str,
                 top10_path: str,
                 reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v2",
                 retriever_model_name: str = "Unknown",
                 reranker_params: Optional[Dict] = None):
        
        logger.info("Initializing evaluator...")
        self.dataset = VietnameseDataset(train_path, articles_path)
        self.retrieval_loader = RetrievalResultsLoader(top10_path)
        
        params = reranker_params or {}
        self.reranker = RerankerModel(
            model_name=reranker_model,
            max_length=params.get('max_length', 512),
            device=params.get('device'),
            trust_remote_code=params.get('trust_remote_code', False),
            default_batch_size=params.get('default_batch_size', 16)
        )
        self.retriever_model_name = retriever_model_name
        self.csv_logger: Optional[RerankerCSVLogger] = None
   
    def run_evaluation(self,
                      batch_size: int = 16,
                      k_values: List[int] = [1, 3, 5, 10],
                      output_file: str = "reranker_results.json",
                      reranked_top10_file: str = "reranked_top10.json",
                      enable_csv_logging: bool = True,
                      csv_base_filename: str = "reranker_eval") -> Dict:
        """Run complete reranking evaluation."""
        
        print("\n" + "="*60)
        print("RERANKER EVALUATION")
        print("="*60)
        stats = self.dataset.get_stats()
        print(f"Questions: {stats['num_questions']}")
        print(f"Documents: {stats['num_documents']}")
        print(f"Retrieval results: {len(self.retrieval_loader)}")
        print("="*60 + "\n")
       
        # Initialize CSV logger
        if enable_csv_logging:
            self.csv_logger = RerankerCSVLogger(csv_base_filename)
       
        # Load model
        print("Loading reranker model...")
        self.reranker.load_model()
       
        # Run reranking
        print("Reranking documents...")
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
       
        # Save results
        if self.csv_logger:
            print("\nSaving CSV logs...")
            self.csv_logger.log_metrics(
                self.reranker.model_name,
                self.retriever_model_name,
                len(self.retrieval_loader),
                metrics
            )
            self.csv_logger.log_detailed_results(sample_results)
            self.csv_logger.log_comparison(sample_results)
       
        print("Saving results...")
        reranked_path = self._save_reranked_top10(sample_results, reranked_top10_file)
        output_path = self._save_results(metrics, sample_results, output_file)
       
        # Display results
        csv_paths = self.csv_logger.get_csv_paths() if self.csv_logger else {}
        self._display_results(metrics, sample_results, output_path, reranked_path, csv_paths)
       
        return {
            "metrics": metrics,
            "sample_results": sample_results,
            "output_file": output_path,
            "reranked_top10_file": reranked_path,
            "csv_files": csv_paths
        }
   
    def _save_reranked_top10(self, sample_results: List[Dict], output_file: str) -> str:
        """Save reranked top-10 documents."""
        reranked_data = []
        for sample in sample_results:
            reranked_data.append({
                "question_id": sample["sample_id"],
                "question": sample["question"],
                "reranked_top_10_documents": sample["reranked_documents"]
            })
       
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reranked_data, f, ensure_ascii=False, indent=2)
       
        logger.info(f"Saved reranked top-10 to {output_path}")
        return str(output_path)
   
    def _save_results(self, metrics: Dict, sample_results: List[Dict], output_file: str) -> str:
        """Save complete results."""
        results = {
            "reranker_model": self.reranker.model_name,
            "retriever_model": self.retriever_model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "dataset_stats": self.dataset.get_stats(),
            "metrics": metrics,
            "sample_results": sample_results
        }
       
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
       
        logger.info(f"Saved results to {output_path}")
        return str(output_path)
   
    def _display_results(self, metrics: Dict, sample_results: List[Dict],
                        output_file: str, reranked_file: str,
                        csv_files: Dict[str, str]) -> None:
        """Display evaluation results."""
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Reranker: {self.reranker.model_name}")
        print(f"Valid samples: {metrics['Valid_Samples']}/{metrics['Total_Samples']}")
        print(f"Avg rerank time: {metrics.get('avg_rerank_time_ms', 0):.2f} ms")
        print(f"Lost relevant docs: {metrics.get('lost_relevant_count', 0)}")
        
        print("\nMetrics:")
        print(f"  MRR: {metrics['MRR']:.4f}")
        for k in [1, 3, 5, 10]:
            if f"Recall@{k}" in metrics:
                print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
        
        # Count improvements
        improved = sum(1 for s in sample_results 
                      if any(d.get('is_relevant') and 
                            d['rank_after_rerank'] < d['rank_before_rerank']
                            for d in s.get('reranked_documents', [])))
        unchanged = sum(1 for s in sample_results 
                       if any(d.get('is_relevant') and 
                             d['rank_after_rerank'] == d['rank_before_rerank']
                             for d in s.get('reranked_documents', [])))
        degraded = sum(1 for s in sample_results 
                      if any(d.get('is_relevant') and 
                            d['rank_after_rerank'] > d['rank_before_rerank']
                            for d in s.get('reranked_documents', [])))
        
        total = improved + unchanged + degraded
        if total > 0:
            print("\nReranking Impact:")
            print(f"  Improved: {improved} ({improved/total*100:.1f}%)")
            print(f"  Unchanged: {unchanged} ({unchanged/total*100:.1f}%)")
            print(f"  Degraded: {degraded} ({degraded/total*100:.1f}%)")
        
        print(f"\nOutput files:")
        print(f"  Results: {output_file}")
        print(f"  Reranked top-10: {reranked_file}")
        
        if csv_files:
            print(f"\nCSV files:")
            for name, path in csv_files.items():
                print(f"  {name}: {path}")
        print("="*60 + "\n")
