import os
import sys
import ctypes

import tiledb

venv_root = os.path.dirname(os.path.dirname(sys.executable))
site_packages = os.path.join(venv_root, 'Lib', 'site-packages')

tiledb_libs = os.path.join(site_packages, 'tiledb.libs')
vector_search_lib = os.path.join(site_packages, 'tiledb', 'vector_search', 'lib')

for directory in [tiledb_libs, vector_search_lib]:
    if os.path.isdir(directory):
        os.add_dll_directory(directory)

if os.path.isdir(tiledb_libs):
    for filename in sorted(os.listdir(tiledb_libs)):
        if filename.endswith('.dll'):
            try:
                ctypes.CDLL(os.path.join(tiledb_libs, filename))
            except Exception:
                pass

if os.path.isdir(vector_search_lib):
    tiledb_dll = os.path.join(vector_search_lib, 'tiledb.dll')
    if os.path.exists(tiledb_dll):
        try:
            ctypes.CDLL(tiledb_dll)
        except Exception:
            pass

import gc
import logging
import threading
import re
import json
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import tiledb.vector_search as vs

from config import get_config
from embedding_models import create_embedding_model
from utilities import get_model_native_precision, get_embedding_dtype_and_batch, configure_logging
from cuda_manager import get_cuda_manager

logger = logging.getLogger(__name__)


def process_chunks_only_query(database_name, query, result_queue):
    configure_logging("INFO")
    try:
        query_db = QueryVectorDB(database_name)
        try:
            contexts, metadata_list = query_db.search(query)

            formatted_contexts = []
            for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
                file_name = metadata.get('file_name', 'Unknown')
                cleaned_context = re.sub(r'\n[ \t]+\n', '\n\n', context)
                cleaned_context = re.sub(r'\n\s*\n\s*\n*', '\n\n', cleaned_context.strip())
                formatted_context = (
                    f"{'-'*80}\n"
                    f"CONTEXT {index} | {file_name}\n"
                    f"{'-'*80}\n"
                    f"{cleaned_context}\n"
                )
                formatted_contexts.append(formatted_context)

            result_queue.put("\n".join(formatted_contexts))
        finally:
            query_db.close()
    except Exception as e:
        result_queue.put(f"Error querying database: {str(e)}")

_thread_local = threading.local()


class QueryVectorDB:
    def __init__(self, selected_database: str):
        self.config = self.load_configuration()

        if not selected_database:
            raise ValueError("No vector database selected.")
        if selected_database not in self.config.created_databases:
            raise ValueError(f'Database "{selected_database}" not found in config.')
        
        db_path = self.config.vector_db_dir / selected_database
        if not db_path.exists():
            raise FileNotFoundError(f'Database folder "{selected_database}" is missing on disk.')

        self.selected_database = selected_database
        self.db_path = db_path
        self.index_uri = str(db_path / "vector_index")
        self.array_uri = str(db_path / "vectors")

        self.embeddings = None
        self.index = None
        self.model_name = None
        self._debug_id = id(self)
        logger.debug(f"Created QueryVectorDB instance {self._debug_id} for database {selected_database}")

        self.distance_metric = "cosine"
        try:
            metadata_file = db_path / "index_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.distance_metric = metadata.get('distance_metric', 'cosine')
                logger.debug(f"Loaded distance metric: {self.distance_metric}")
        except Exception as e:
            logger.warning(f"Could not load index metadata, using default cosine metric: {e}")

    def load_configuration(self):
        try:
            return get_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    @torch.inference_mode()
    def initialize_vector_model(self):
        model_path = self.config.created_databases[self.selected_database].model
        self.model_name = os.path.basename(model_path)
        
        compute_device = self.config.Compute_Device.database_query
        use_half = self.config.database.half
        model_native_precision = get_model_native_precision(self.model_name)

        dtype, batch_size = get_embedding_dtype_and_batch(
            compute_device=compute_device,
            use_half=use_half,
            model_native_precision=model_native_precision,
            model_name=self.model_name,
            is_query=True
        )

        return create_embedding_model(
            model_path=model_path,
            compute_device=compute_device,
            dtype=dtype,
            batch_size=batch_size,
            is_query=True
        )

    @torch.inference_mode()
    def search(self, query, k: Optional[int] = None, score_threshold: Optional[float] = None):
        cuda_mgr = get_cuda_manager()

        if not self.embeddings:
            logger.info(f"Initializing embedding model for database {self.selected_database}")
            self.embeddings = self.initialize_vector_model()

        if not self.index:
            logger.info(f"Loading TileDB index for {self.selected_database}")
            self.index = vs.FlatIndex(uri=self.index_uri)

        self.config = self.load_configuration()
        k = k if k is not None else self.config.database.contexts
        score_threshold = score_threshold if score_threshold is not None else self.config.database.similarity

        with cuda_mgr.cuda_operation():
            query_vector = self.embeddings.embed_query(query)

        query_vector_np = np.array([query_vector], dtype=np.float32)

        logger.info(f"Querying TileDB index: {self.index_uri}")

        result_distances, result_ids = self.index.query(
            query_vector_np,
            k=k
        )

        if len(result_distances) == 0 or len(result_distances[0]) == 0:
            logger.warning("No results returned from vector search")
            return [], []

        distances = result_distances[0]
        ids = result_ids[0]

        if self.distance_metric == "cosine":
            similarities = 1 - distances
            similarities = np.clip(similarities, 0.0, 1.0)

        elif self.distance_metric == "l2":
            if len(distances) > 0 and np.max(distances) > 0:
                similarities = 1 - (distances / np.max(distances))
            else:
                similarities = np.ones_like(distances)
            similarities = np.clip(similarities, 0.0, 1.0)

        elif self.distance_metric == "sum_of_squares":
            if len(distances) > 0 and np.max(distances) > 0:
                similarities = 1 - (distances / np.max(distances))
            else:
                similarities = np.ones_like(distances)
            similarities = np.clip(similarities, 0.0, 1.0)

        elif self.distance_metric == "inner_product":
            if len(distances) > 0:
                min_val = np.min(distances)
                max_val = np.max(distances)
                if max_val > min_val:
                    similarities = (distances - min_val) / (max_val - min_val)
                else:
                    similarities = np.ones_like(distances)
            else:
                similarities = np.array([])

        else:
            logger.warning(f"Unknown distance metric '{self.distance_metric}', using default conversion")
            similarities = 1 - (distances ** 2) / 2
            similarities = np.clip(similarities, 0.0, 1.0)

        results = []

        valid_indices = similarities >= score_threshold
        if not np.any(valid_indices):

            pass
        else:
            filtered_distances = distances[valid_indices]
            filtered_ids = ids[valid_indices]
            filtered_similarities = similarities[valid_indices]

            with tiledb.open(self.array_uri, mode='r') as A:
                data = A.multi_index[filtered_ids.astype(np.uint64)]

                texts_raw = data['text']
                metadatas_raw = data['metadata']

                for i, (distance, vec_id, similarity) in enumerate(zip(filtered_distances, filtered_ids, filtered_similarities)):
                    try:
                        text_raw = texts_raw[i]
                        if isinstance(text_raw, np.ndarray):
                            text = text_raw.item() if text_raw.size == 1 else str(text_raw[0])
                        else:
                            text = str(text_raw)

                        metadata_raw = metadatas_raw[i]
                        if isinstance(metadata_raw, np.ndarray):
                            metadata_str = metadata_raw.item() if metadata_raw.size == 1 else str(metadata_raw[0])
                        else:
                            metadata_str = str(metadata_raw)

                        metadata = json.loads(metadata_str)
                        metadata['similarity_score'] = float(similarity)
                        metadata['distance'] = float(distance)
                        results.append((text, metadata))

                    except json.JSONDecodeError as je:
                        logger.warning(f"Failed to parse JSON for vector ID {vec_id}: {je}")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to retrieve data for vector ID {vec_id}: {e}")
                        continue

        search_term = self.config.database.search_term.lower()
        if search_term:
            filtered_results = [
                (text, metadata) for text, metadata in results
                if search_term in text.lower()
            ]
        else:
            filtered_results = results

        document_types = self.config.database.document_types
        if document_types:
            filtered_results = [
                (text, metadata) for text, metadata in filtered_results
                if metadata.get('document_type') == document_types
            ]

        contexts = [text for text, _ in filtered_results]
        metadata_list = [metadata for _, metadata in filtered_results]

        logger.info(f"Found {len(contexts)} matching contexts")
        return contexts, metadata_list

    def close(self):
        logger.info(f"Closing QueryVectorDB instance {self._debug_id} for database {self.selected_database}")

        if self.embeddings:
            logger.debug(f"Unloading embedding model for database {self.selected_database}")
            del self.embeddings
            self.embeddings = None

        if self.index:
            del self.index
            self.index = None

        get_cuda_manager().safe_empty_cache()
        gc.collect()
        logger.debug(f"Cleanup completed for instance {self._debug_id}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def get_query_db(database_name: str) -> QueryVectorDB:
    if not hasattr(_thread_local, 'query_db_cache'):
        _thread_local.query_db_cache = {}
    
    if database_name in _thread_local.query_db_cache:
        cached_db = _thread_local.query_db_cache[database_name]
        logger.debug(f"Reusing thread-local QueryVectorDB for {database_name}")
        return cached_db

    logger.debug(f"Creating new thread-local QueryVectorDB for {database_name}")
    db_instance = QueryVectorDB(database_name)
    _thread_local.query_db_cache[database_name] = db_instance
    return db_instance


def clear_query_cache(database_name: Optional[str] = None):
    if not hasattr(_thread_local, 'query_db_cache'):
        return
    
    if database_name:
        if database_name in _thread_local.query_db_cache:
            logger.info(f"Clearing thread-local cache for {database_name}")
            _thread_local.query_db_cache[database_name].close()
            del _thread_local.query_db_cache[database_name]
    else:
        logger.info("Clearing all thread-local query database cache")
        for db_name, db_instance in _thread_local.query_db_cache.items():
            db_instance.close()
        _thread_local.query_db_cache.clear()