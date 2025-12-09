"""
Entity extraction and linking for graph population.

In production, use:
- spaCy NER for general entities
- GPT-4 function calling for structured extraction
- Domain-specific entity recognition models
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .graph_builder import Entity

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0


class EntityExtractor:
    """
    Extract entities from text for graph population.

    This is a simplified implementation. For production, use:
    - spaCy for NER
    - OpenAI function calling for structured extraction
    - Domain-specific entity recognition

    Usage:
        extractor = EntityExtractor()
        entities = extractor.extract("Acme Corp signed contract C-2024-001")
    """

    def __init__(self, entity_patterns: Dict[str, List[str]] = None):
        """
        Args:
            entity_patterns: {type: [keyword patterns]}
        """
        self.entity_patterns = entity_patterns or {
            "Customer": [
                "customer",
                "client",
                "buyer",
                "company",
                "corp",
                "inc",
                "ltd",
            ],
            "Contract": ["contract", "agreement", "deal", "c-\\d+"],
            "Product": ["product", "service", "solution", "software", "platform"],
            "Region": [
                "region",
                "territory",
                "area",
                "apac",
                "emea",
                "americas",
                "latam",
            ],
            "Person": ["mr", "ms", "dr", "ceo", "cto", "manager", "director"],
        }

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text.

        Args:
            text: Document text

        Returns:
            List of extracted entities
        """
        entities = []
        text_lower = text.lower()

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                # Use regex for pattern matching
                regex = re.compile(rf"\b{pattern}\b", re.IGNORECASE)
                for match in regex.finditer(text):
                    entities.append(
                        ExtractedEntity(
                            text=match.group(),
                            type=entity_type,
                            start=match.start(),
                            end=match.end(),
                        )
                    )

        # Deduplicate by text
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.type)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    def extract_with_context(
        self,
        text: str,
        context_window: int = 50,
    ) -> List[Tuple[ExtractedEntity, str]]:
        """
        Extract entities with surrounding context.

        Args:
            text: Document text
            context_window: Characters of context to include

        Returns:
            List of (entity, context) tuples
        """
        entities = self.extract(text)
        results = []

        for entity in entities:
            start = max(0, entity.start - context_window)
            end = min(len(text), entity.end + context_window)
            context = text[start:end]
            results.append((entity, context))

        return results


class EntityLinker:
    """
    Link extracted entities to graph nodes.

    Handles entity resolution/disambiguation:
    - "Acme" and "Acme Corp" -> same entity
    - Context-based disambiguation

    Usage:
        linker = EntityLinker(graph)
        linked = linker.link(extracted_entities)
    """

    def __init__(self, graph=None):
        """
        Args:
            graph: Knowledge graph for linking
        """
        self.graph = graph
        self._alias_map: Dict[str, str] = {}  # alias -> canonical entity ID

    def add_alias(self, alias: str, entity_id: str) -> None:
        """Add an alias for an entity."""
        self._alias_map[alias.lower()] = entity_id

    def link(
        self,
        extracted: List[ExtractedEntity],
        create_if_missing: bool = True,
    ) -> List[Tuple[ExtractedEntity, Optional[Entity]]]:
        """
        Link extracted entities to graph nodes.

        Args:
            extracted: Extracted entities
            create_if_missing: Create new graph entities if not found

        Returns:
            List of (extracted, linked_entity) tuples
        """
        results = []

        for ext in extracted:
            # Try alias lookup
            entity_id = self._alias_map.get(ext.text.lower())

            if entity_id and self.graph:
                entity = self.graph.get_entity(entity_id)
                if entity:
                    results.append((ext, entity))
                    continue

            # Try fuzzy match on graph entities
            if self.graph:
                match = self._fuzzy_match(ext)
                if match:
                    results.append((ext, match))
                    continue

            # Create new entity if allowed
            if create_if_missing:
                new_entity = Entity(
                    id=f"{ext.type.lower()}_{ext.text.lower().replace(' ', '_')}",
                    type=ext.type,
                    name=ext.text.title(),
                )
                if self.graph:
                    self.graph.add_entity(new_entity)
                results.append((ext, new_entity))
            else:
                results.append((ext, None))

        return results

    def _fuzzy_match(
        self,
        extracted: ExtractedEntity,
        threshold: float = 0.8,
    ) -> Optional[Entity]:
        """
        Fuzzy match extracted entity to graph.

        Uses simple string similarity.
        For production, use embedding similarity or learned models.
        """
        if not self.graph:
            return None

        candidates = self.graph.get_entities_by_type(extracted.type)

        best_match = None
        best_score = 0

        for candidate in candidates:
            score = self._similarity(extracted.text.lower(), candidate.name.lower())
            if score > threshold and score > best_score:
                best_score = score
                best_match = candidate

        return best_match

    def _similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity (Jaccard on characters)."""
        set1 = set(s1)
        set2 = set(s2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union
