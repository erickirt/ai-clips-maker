"""
Embed text using the Roberta model for downstream segmentation tasks.
"""

import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Generates vector representations (embeddings) of text using the 'all-roberta-large-v1' model.
    Useful for semantic comparison and text segmentation.
    """

    def __init__(self) -> None:
        """
        Initializes the SentenceTransformer model.
        """
        self.__model = SentenceTransformer("all-roberta-large-v1")

    def embed_sentences(self, sentences: list[str]) -> torch.Tensor:
        """
        Transforms a list of sentences into embedding vectors.

        Parameters
        ----------
        sentences: list[str]
            A list of strings, where each string is a sentence.

        Returns
        -------
        torch.Tensor
            A 2D tensor of shape (N x E), where N is the number of sentences
            and E is the embedding dimension.
        """
        embeddings = self.__model.encode(sentences, convert_to_tensor=True)
        return embeddings
