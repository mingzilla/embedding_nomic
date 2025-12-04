import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict
from torch import nn
from sentence_transformers import SentenceTransformer, models

from shared_utils.external.json_config.json_config_util import JsonConfigUtil

logger = logging.getLogger(__name__)


class LastTokenPooler(nn.Module):
    """
    A custom pooling module that takes the embedding of the last token.
    This is the correct strategy for decoder-only models like Gemma.
    """
    def __init__(self, word_embedding_dimension: int):
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension

    def forward(self, features: Dict[str, torch.Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        # Find the length of each sequence by counting non-padding tokens
        seq_lengths = attention_mask.sum(dim=1) - 1
        
        # Get the embedding of the last token for each sequence
        batch_size = token_embeddings.shape[0]
        batch_indices = torch.arange(batch_size, device=token_embeddings.device)
        last_token_embeddings = token_embeddings[batch_indices, seq_lengths]
        
        # Add the new key to the output dictionary
        features['sentence_embedding'] = last_token_embeddings
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension


class ModelConfig:
    def __init__(self, config_name: str, config_data: dict):
        self.config_name = config_name
        self.model_name = config_data["model"]
        self.docker_image_name = config_data["docker-image-name"]
        self.docker_image_version = config_data["docker-image-version"]
        self.features = config_data["features"]
        self.trust_remote_code = config_data["trust_remote_code"]
        self.max_input_chars = config_data.get("max_input_chars")
        self.matryoshka_dims = config_data.get("matryoshka_dims")

    def supports_matryoshka(self) -> bool:
        return self.matryoshka_dims is not None and len(self.matryoshka_dims) > 0

    def validate_dimension(self, dimension: int) -> bool:
        if not self.supports_matryoshka():
            return False
        return dimension in self.matryoshka_dims


class ModelLoader:
    _instance: Optional['ModelLoader'] = None
    _model: Optional[SentenceTransformer] = None
    _config: Optional[ModelConfig] = None
    _device: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        config_name = os.getenv("CONFIG_NAME")
        if not config_name:
            raise ValueError("CONFIG_NAME environment variable not set")

        build_config_path = Path(__file__).parent.parent / "_config-definition" / "build-config.json"
        logger.info(f"Loading build configuration from: {build_config_path}")

        build_config = JsonConfigUtil.read(str(build_config_path))

        if config_name not in build_config:
            available_keys = ", ".join(build_config.keys())
            raise ValueError(
                f"Invalid CONFIG_NAME '{config_name}'. "
                f"Available keys: {available_keys}"
            )

        model_config_data = build_config[config_name]
        self._config = ModelConfig(config_name, model_config_data)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self._device}")

        if self._device == "cpu":
            logger.warning("CUDA not available! Running on CPU will be significantly slower.")

        # 1. Load the base transformer model.
        logger.info(f"Loading base model '{self._config.model_name}'...")
        word_embedding_model = models.Transformer(
            self._config.model_name,
            model_args={'trust_remote_code': self._config.trust_remote_code},
        )
        logger.info("Base model loaded successfully.")

        # 2. Decide which pooling module to use based on model architecture.
        modules = [word_embedding_model]
        model_name_lower = self._config.model_name.lower()
        embedding_dimension = word_embedding_model.get_word_embedding_dimension()

        if 'gemma' in model_name_lower:
            logger.info("Gemma (Decoder-only) model detected. Adding LastTokenPooler.")
            modules.append(LastTokenPooler(embedding_dimension))
        else:
            logger.info("Encoder model detected. Adding standard MEAN pooling layer.")
            pooling_model = models.Pooling(embedding_dimension)
            modules.append(pooling_model)

        # 3. Create the final SentenceTransformer model.
        self._model = SentenceTransformer(modules=modules, device=self._device)

        # 5. Get model dimension for logging.
        model_dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model ready on {self._device}")
        logger.info(f"  Full embedding dimension: {model_dimension}")

        if self._config.supports_matryoshka():
            logger.info(f"  Matryoshka dimensions: {self._config.matryoshka_dims}")

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        instance = cls()
        if instance._model is None:
            raise RuntimeError("Model not loaded")
        return instance._model

    @classmethod
    def get_config(cls) -> ModelConfig:
        instance = cls()
        if instance._config is None:
            raise RuntimeError("Config not loaded")
        return instance._config

    @classmethod
    def get_device(cls) -> str:
        instance = cls()
        if instance._device is None:
            raise RuntimeError("Device not set")
        return instance._device
