import spacy
from typing import List, Dict, Tuple, Any


class NamedEntityService:
    """
    A class for extracting named entities from text and using them for enhanced document searching.
    """

    def __init__(self, model_name: str = "en_core_web_sm", k: int = 4):
        """
        Initialize the EntityExtractor with a spaCy model.

        Args:
            model_name: The name of the spaCy model to use.
        """
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from the given text.

        Args:
            text: The text to extract entities from.

        Returns:
            A list of dictionaries containing entity information.
        """
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )

        return entities

    def search_chunks_for_entities(
        self,
        query: str,
        chunks: List[str],
        k: int = 4,
        entity_weights: Dict[str, float] = {
            "PERSON": 2.0,
            "ORG": 1.8,
            "GPE": 1.5,  # Geopolitical Entity (countries, cities, etc.)
            "DATE": 1.3,
            "TIME": 1.3,
            "MONEY": 1.2,
            "PRODUCT": 1.5,
            "EVENT": 1.4,
            "DEFAULT": 1.0,  # Default weight for other entity types
        },
    ) -> List[Tuple[int, float, str]]:
        """
        Search document chunks for entities found in the query.

        Args:
            query: The user query to extract entities from.
            chunks: List of document chunks to search through.
            entity_weights: Optional dictionary mapping entity types to weights.
                           Default weights prioritize PERSON, ORG, and GPE entities.

        Returns:
            List of tuples containing (chunk_index, score, chunk_text).
        """

        # Extract entities from query
        entities = self.extract_entities(query)

        # Score each chunk based on entity matches
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            score = 0
            chunk_lower = chunk.lower()

            for entity in entities:
                entity_text = entity["text"].lower()
                entity_type = entity["label"]

                # Check if entity appears in the chunk
                if entity_text in chunk_lower:
                    # Apply weight based on entity type
                    weight = entity_weights.get(entity_type, entity_weights["DEFAULT"])
                    score += weight

                    # Bonus for exact case match (might indicate proper names)
                    if entity["text"] in chunk:
                        score += 0.2

            chunk_scores.append((i, score, chunk))

        # Sort chunks by score (descending)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        res = [chunk for i, score, chunk in chunk_scores]
        scores = [score for i, score, chunk in chunk_scores]

        return res[:k], scores[:k]
