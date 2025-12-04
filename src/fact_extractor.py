"""
Fact Extractor
Extracts new facts from conversations and checks if they're already in the index.
"""

import hashlib
import logging
from typing import List, Dict, Any, Set
from .litellm_client import AsyncLiteLLMClient
from .rag_search import RAGSearch

logger = logging.getLogger(__name__)


class FactExtractor:
    """
    Extracts candidate facts from conversations and validates them against the index.
    """
    
    def __init__(self, client: AsyncLiteLLMClient, rag_search: RAGSearch):
        """
        Initialize fact extractor.
        
        Args:
            client: AsyncLiteLLMClient instance
            rag_search: RAGSearch instance for index queries
        """
        self.client = client
        self.rag_search = rag_search
    
    def _normalize_fact(self, fact_text: str) -> str:
        """
        Normalize fact text for hashing/deduplication.
        
        Args:
            fact_text: Original fact text
            
        Returns:
            Normalized text
        """
        return fact_text.strip().lower()
    
    def _hash_fact(self, fact_text: str) -> str:
        """
        Generate hash for fact deduplication.
        
        Args:
            fact_text: Fact text
            
        Returns:
            Hash string
        """
        normalized = self._normalize_fact(fact_text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    async def extract_candidate_facts(
        self,
        conversation_history: List[Dict[str, str]],
        used_search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract candidate facts from conversation that are not in used search results.
        
        Args:
            conversation_history: Full conversation history
            used_search_results: All chunks used in RAG answers for this session
            
        Returns:
            List of candidate fact strings
        """
        logger.info(f"[FACT-EXTRACTOR] Extracting candidate facts - History: {len(conversation_history)} messages, Used results: {len(used_search_results)} chunks")
        
        # Build conversation text
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        # Build used search results text
        used_results_text = "\n\n".join([
            chunk["text"] for chunk in used_search_results
        ])
        
        logger.debug(f"[FACT-EXTRACTOR] Conversation text length: {len(conversation_text)} chars, Used results text length: {len(used_results_text)} chars")
        
        prompt = f"""You are analyzing a conversation about Mass Effect lore to extract new factual statements that are not already present in the knowledge base.

CONVERSATION HISTORY:
{conversation_text}

KNOWLEDGE BASE CONTENT (already used in answers):
{used_results_text}

Your task: Extract factual statements from the conversation that:
1. Are about Mass Effect lore (characters, events, locations, technology, history, factions, etc.)
2. Are explicitly stated in the conversation (not invented or inferred)
3. Are NOT present in the knowledge base content above (even if paraphrased)
4. Are not questions, opinions, or speculation
5. Are clear, specific, and self-contained factual statements
6. Are genuinely new information relative to the knowledge base

Output each fact on a separate line. Each fact should be a complete, standalone statement.
If no new facts are found, output only "NONE".
"""
        
        logger.info(f"[FACT-EXTRACTOR] Calling GPT-5-nano to extract candidate facts")
        response = await self.client.text_completion(
            prompt=prompt,
            model_name="azure.gpt-5-nano",
            max_tokens=20000,
            temperature=0.3
        )
        
        # Parse facts
        facts = [line.strip() for line in response.strip().split('\n') if line.strip() and line.strip().upper() != "NONE"]
        
        logger.info(f"[FACT-EXTRACTOR] Extracted {len(facts)} candidate facts")
        for i, fact in enumerate(facts, 1):
            logger.info(f"[FACT-EXTRACTOR] Candidate fact {i}/{len(facts)}: {fact[:100]}...")
        
        return facts
    
    async def check_index_presence(self, fact_text: str) -> bool:
        """
        Check if a fact is already present in the index using top-10 nearest + GPT-5-nano boolean check.
        
        Args:
            fact_text: Fact text to check
            
        Returns:
            True if fact is already in index, False otherwise
        """
        logger.info(f"[FACT-EXTRACTOR] Checking index presence for fact: {fact_text[:80]}...")
        
        # Get top-10 nearest chunks
        nearest_chunks = await self.rag_search.vector_search(fact_text, k=10)
        logger.info(f"[FACT-EXTRACTOR] Found {len(nearest_chunks)} nearest chunks for fact check")
        
        # Build context from nearest chunks
        chunk_texts = [chunk["text"] for chunk in nearest_chunks]
        context_text = "\n\n".join(chunk_texts)
        
        top_similarities = [f"{chunk['similarity']:.4f}" for chunk in nearest_chunks[:3]]
        logger.debug(f"[FACT-EXTRACTOR] Top chunk similarities: {top_similarities}")
        
        prompt = f"""You are checking if a factual statement is already present in a knowledge base.

FACT TO CHECK:
{fact_text}

KNOWLEDGE BASE CONTENT (10 most similar chunks):
{context_text}

Question: Does the information in these knowledge base chunks already contain this fact, even if paraphrased or expressed differently?

Answer with only "true" or "false" (lowercase, no punctuation, no explanation).
"""
        
        logger.debug(f"[FACT-EXTRACTOR] Calling GPT-5-nano to check if fact is already in index")
        response = await self.client.text_completion(
            prompt=prompt,
            model_name="azure.gpt-5-nano",
            max_tokens=20000,
            temperature=0.3
        )
        
        # Parse boolean response
        response_lower = response.strip().lower()
        is_present = response_lower.startswith("true")
        logger.info(f"[FACT-EXTRACTOR] Index presence check result: {'PRESENT' if is_present else 'NEW'} (response: {response_lower[:20]})")
        return is_present
    
    async def process_facts(
        self,
        conversation_history: List[Dict[str, str]],
        used_search_results: List[Dict[str, Any]],
        seen_fact_hashes: Set[str] = None
    ) -> List[str]:
        """
        Complete fact processing pipeline: extract candidates and check against index.
        
        Args:
            conversation_history: Full conversation history
            used_search_results: All chunks used in RAG answers
            seen_fact_hashes: Set of fact hashes already seen (for deduplication)
            
        Returns:
            List of new facts that passed all checks
        """
        logger.info(f"[FACT-EXTRACTOR] Starting fact processing pipeline (seen_hashes: {len(seen_fact_hashes) if seen_fact_hashes else 0})")
        
        if seen_fact_hashes is None:
            seen_fact_hashes = set()
        
        # Step 1: Extract candidate facts
        logger.info(f"[FACT-EXTRACTOR] Step 1: Extracting candidate facts")
        candidate_facts = await self.extract_candidate_facts(
            conversation_history,
            used_search_results
        )
        
        if not candidate_facts:
            logger.info(f"[FACT-EXTRACTOR] No candidate facts extracted")
            return []
        
        # Step 2: Deduplicate against seen facts
        logger.info(f"[FACT-EXTRACTOR] Step 2: Deduplicating {len(candidate_facts)} candidate facts against {len(seen_fact_hashes)} seen hashes")
        new_candidates = []
        for fact in candidate_facts:
            fact_hash = self._hash_fact(fact)
            if fact_hash not in seen_fact_hashes:
                new_candidates.append(fact)
                seen_fact_hashes.add(fact_hash)
                logger.debug(f"[FACT-EXTRACTOR] New candidate fact (hash: {fact_hash[:8]}...): {fact[:80]}...")
            else:
                logger.debug(f"[FACT-EXTRACTOR] Duplicate fact skipped (hash: {fact_hash[:8]}...): {fact[:80]}...")
        
        logger.info(f"[FACT-EXTRACTOR] Step 2: {len(new_candidates)} new candidates after deduplication (removed {len(candidate_facts) - len(new_candidates)} duplicates)")
        
        if not new_candidates:
            logger.info(f"[FACT-EXTRACTOR] No new candidates after deduplication")
            return []
        
        # Step 3: Check each candidate against index
        logger.info(f"[FACT-EXTRACTOR] Step 3: Checking {len(new_candidates)} candidates against index")
        new_facts = []
        for i, fact in enumerate(new_candidates, 1):
            logger.info(f"[FACT-EXTRACTOR] Checking candidate {i}/{len(new_candidates)}")
            is_present = await self.check_index_presence(fact)
            if not is_present:
                new_facts.append(fact)
                logger.info(f"[FACT-EXTRACTOR] Candidate {i} is NEW - added to final list")
            else:
                logger.info(f"[FACT-EXTRACTOR] Candidate {i} is already in index - skipped")
        
        logger.info(f"[FACT-EXTRACTOR] Fact processing complete: {len(new_facts)} new facts found out of {len(candidate_facts)} candidates")
        return new_facts

