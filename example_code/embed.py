"""
Batch Embedding Entry Point

This script loads the sentence-transformers model and processes a DuckDB file
to generate embeddings using GPU acceleration.

The core DuckDB I/O logic is delegated to the reusable processor module,
making this script a thin wrapper that only handles model-specific concerns.

Usage:
    python -m embed

Requirements:
    - embedding_config.json must exist in /app/data/ (mounted volume in Docker)
    - Input DuckDB file must exist at the path specified in embedding_config.json
"""

import os
import logging

from shared_utils.external.embed_with_duckdb_io.embedding_processor import EmbeddingProcessor, EmbeddingConfig
from model_loader import ModelLoader


def main():
    """Main entry point for batch embedding process."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting Batch Embedding Process")
    logger.info("=" * 80)

    # Load configuration from environment variable
    config_path = os.getenv("CONFIG_PATH", "/app/data/embedding_config.json")
    logger.info(f"Loading configuration from: {config_path}")

    try:
        config = EmbeddingConfig(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please ensure embedding_config.json is in the mounted data directory")
        raise
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        raise

    # Prepend /app/data/ to relative paths (user shouldn't know about Docker internals)
    data_dir = os.getenv("APP_DATA_PATH", "/app/data/_app_data")
    if not os.path.isabs(config.input_db_path):
        config.input_db_path = os.path.join(data_dir, config.input_db_path)
    if not os.path.isabs(config.output_db_path):
        config.output_db_path = os.path.join(data_dir, config.output_db_path)

    logger.info(f"Configuration loaded successfully")
    logger.info(f"  Input DB: {config.input_db_path}")
    logger.info(f"  Output DB: {config.output_db_path}")
    logger.info(f"  Input Table: {config.input_table}")
    logger.info(f"  Output Table: {config.output_table}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Total Rows Limit: {config.total_rows}")
    logger.info(f"  Embedding Dimension: {config.embedding_dimension}")

    # Load model using ModelLoader
    model = ModelLoader.get_model()
    model_config = ModelLoader.get_config()
    device = ModelLoader.get_device()

    # Get and log the actual model embedding dimension
    model_dimension = model.get_sentence_embedding_dimension()
    logger.info(f"Model full dimension: {model_dimension}")
    logger.info(f"Requested dimension: {config.embedding_dimension}")

    # Determine effective character limit
    runtime_input_chars = getattr(config, 'input_chars', None)
    if runtime_input_chars is not None:
        if model_config.max_input_chars and runtime_input_chars > model_config.max_input_chars:
            raise ValueError(
                f"Runtime input_chars ({runtime_input_chars}) exceeds build-time "
                f"max_input_chars ({model_config.max_input_chars}) for model {model_config.config_name}"
            )
        effective_char_limit = runtime_input_chars
        logger.info(f"Using runtime input_chars: {effective_char_limit}")
    else:
        effective_char_limit = model_config.max_input_chars
        logger.info(f"Using build-time max_input_chars: {effective_char_limit}")

    # Validate dimension for Matryoshka models
    if model_config.supports_matryoshka():
        if config.embedding_dimension == model_dimension:
            logger.info(f"Using full dimension (no truncation)")
        elif model_config.validate_dimension(config.embedding_dimension):
            logger.info(
                f"Using Matryoshka truncation: {model_dimension}d -> {config.embedding_dimension}d"
            )
        else:
            raise ValueError(
                f"Invalid embedding_dimension {config.embedding_dimension} for {model_config.config_name}. "
                f"Allowed dimensions: {model_config.matryoshka_dims}"
            )
    else:
        if config.embedding_dimension != model_dimension:
            raise ValueError(
                f"Model {model_config.config_name} does not support Matryoshka. "
                f"embedding_dimension must be {model_dimension}, got {config.embedding_dimension}"
            )

    # Define embedding callback function
    # This encapsulates all model-specific logic
    def embed_texts_callback(texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts, with pre-truncation.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        # For performance, pre-truncate texts to the effective character limit.
        # The model's tokenizer will still perform its own token-based truncation,
        # but this avoids processing excessively long strings in the Python layer.
        if effective_char_limit and effective_char_limit > 0:
            processed_texts = [text[:effective_char_limit] for text in texts]
        else:
            processed_texts = texts

        embeddings = model.encode(
            processed_texts,
            convert_to_tensor=False, # True to use GPU memory. Good if we need to use GPU as next step. Bad if the next step is a CPU task
            normalize_embeddings=True,
            # batch_size=128, # this can improve performance, 100M records costs from 40s to 30s - note: high value with too much input can crash machine
            show_progress_bar=False
        )

        # Apply Matryoshka truncation if needed
        if model_config.supports_matryoshka() and config.embedding_dimension < model_dimension:
            embeddings = embeddings[:, :config.embedding_dimension]

        return embeddings.tolist()

    # Delegate to the reusable DuckDB processor
    logger.info("Starting DuckDB processing...")
    try:
        EmbeddingProcessor.process_duckdb(config, embed_texts_callback)
        logger.info("Batch embedding process completed successfully!")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
