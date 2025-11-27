import logging
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

from config import get_config
from utilities_core import (
    supports_flash_attention,
    get_embedding_dtype_and_batch,
    get_model_native_precision
)

logger = logging.getLogger(__name__)


def _get_model_family(model_path: str) -> str:
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower or "qwen3-embedding" in model_path_lower:
        return "qwen"
    elif "bge" in model_path_lower:
        return "bge"
    else:
        return "generic"


def _get_prompt_for_family(family: str, is_query: bool = False) -> str:
    if family == "qwen" and is_query:
        return "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    elif family == "bge":
        return "Represent this sentence for searching relevant passages: "
    else:
        return ""


def _validate_and_clean_texts(texts: list) -> list[str]:
    for idx, text in enumerate(texts):
        if text is None:
            texts[idx] = " "
            logger.warning(f"Replaced None with space at index {idx}")
        elif isinstance(text, (list, tuple)):
            texts[idx] = " ".join(str(item) for item in text if item)
            logger.warning(f"Flattened {type(text).__name__} to string at index {idx}")
        elif not isinstance(text, str):
            texts[idx] = str(text)
            logger.warning(f"Converted {type(text).__name__} to string at index {idx}")

        if isinstance(texts[idx], str):
            texts[idx] = texts[idx].strip() or " "

    return texts


class DirectEmbeddingModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = None,
        batch_size: int = 8,
        max_seq_length: int = 512,
        prompt: str = "",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.prompt = prompt
        self.model = None

        logger.info(f"Initializing DirectEmbeddingModel: {os.path.basename(model_path)}")
        self._initialize_model()

    def _initialize_model(self):
        family = _get_model_family(self.model_path)

        model_kwargs = {
            'torch_dtype': self.dtype if self.dtype else torch.float32,
        }

        is_cuda = self.device.lower().startswith("cuda")
        if family == "qwen":
            if is_cuda and supports_flash_attention():
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.debug("Using flash_attention_2 for Qwen model")
            else:
                model_kwargs['attn_implementation'] = 'sdpa'
                logger.debug("Using sdpa for Qwen model")
        else:
            model_kwargs['attn_implementation'] = 'sdpa'

        tokenizer_kwargs = {
            'model_max_length': self.max_seq_length,
        }

        if family == "qwen":
            tokenizer_kwargs['padding_side'] = 'left'

        logger.info("Loading SentenceTransformer model...")
        self.model = SentenceTransformer(
            model_name_or_path=self.model_path,
            device=self.device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        self.model.to(self.device)

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"  - Dtype: {self.dtype}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Max sequence length: {self.max_seq_length}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        encode_kwargs = {
            'batch_size': self.batch_size,
            'normalize_embeddings': True,
            'convert_to_numpy': True,
            'show_progress_bar': True,
            'device': self.device,
        }

        logger.debug(f"Embedding {len(texts)} documents...")

        try:
            embeddings = self.model.encode(texts, **encode_kwargs)
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            logger.error("Falling back to single-text encoding...")

            all_embeddings = []
            for idx, text in enumerate(texts):
                try:
                    single_emb = self.model.encode(
                        [text],
                        batch_size=1,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        device=self.device,
                    )
                    all_embeddings.append(single_emb[0].tolist())
                except Exception as e_single:
                    preview = repr(text)
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    logger.error(f"Skipping text at index {idx}: {e_single}")
                    logger.error(f"Problematic text preview: {preview}")

            return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        if self.prompt:
            text = self.prompt + text

        encode_kwargs = {
            'batch_size': 1,
            'normalize_embeddings': True,
            'convert_to_numpy': True,
            'show_progress_bar': False,
            'device': self.device,
        }

        embedding = self.model.encode(
            [text],
            **encode_kwargs
        )

        return embedding[0].tolist()

    def __del__(self):
        if self.model is not None:
            del self.model
            self.model = None


def create_embedding_model(
    model_path: str,
    compute_device: str = "cpu",
    dtype: torch.dtype = None,
    batch_size: int = None,
    is_query: bool = False,
) -> DirectEmbeddingModel:

    config = get_config()
    model_name = os.path.basename(model_path)

    family = _get_model_family(model_path)
    model_native_precision = get_model_native_precision(model_name)

    use_half = config.database.half
    _dtype, _batch_size = get_embedding_dtype_and_batch(
        compute_device=compute_device,
        use_half=use_half,
        model_native_precision=model_native_precision,
        model_name=model_name,
        is_query=is_query
    )

    final_dtype = dtype if dtype is not None else _dtype
    final_batch_size = batch_size if batch_size is not None else _batch_size

    if family == "qwen":
        max_seq_length = 8192
    else:
        max_seq_length = 512

    prompt = _get_prompt_for_family(family, is_query)

    logger.info(f"Creating embedding model: {model_name}")
    logger.info(f"  - Family: {family}")
    logger.info(f"  - Device: {compute_device}")
    logger.info(f"  - Dtype: {final_dtype}")
    logger.info(f"  - Batch size: {final_batch_size}")
    logger.info(f"  - Max sequence: {max_seq_length}")
    if prompt:
        logger.info(f"  - Using prompt: {prompt[:50]}...")

    return DirectEmbeddingModel(
        model_path=model_path,
        device=compute_device,
        dtype=final_dtype,
        batch_size=final_batch_size,
        max_seq_length=max_seq_length,
        prompt=prompt,
    )