"""
llm_config.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-03-2025
"""

# Standard library imports
from typing import Optional
import os

# Third-party imports
from crewai import LLM

# Custom imports
from ..logger import setup_logger

######## logger Configuration ########
logger = setup_logger("composition_property_extractor.log")


class LLMConfig:
    # Default values
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TOP_P = 0.9
    DEFAULT_TIMEOUT = 60
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.api_base = api_base
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature or self.DEFAULT_TEMPERATURE
        self.top_p = top_p or self.DEFAULT_TOP_P
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens

    def get_llm(self) -> LLM:
        """
        Creates and returns an LLM instance based on the model configuration
        for standard CrewAI supported providers.
        """
        # Initialize kwargs with non-None values
        kwargs = {k: v for k, v in self.__dict__.items() if v is not None}

        # Remove model from kwargs as it's handled separately
        kwargs.pop("model", None)

        # Create LLM instance with standard configurations
        standard_kwargs = {"model": self.model}
        standard_kwargs.update(kwargs)
        return LLM(**standard_kwargs)
