"""
LLM-Based Tag Extractor for Trajectories.

Extracts semantic tags from trajectory takeaways using an LLM.
"""

import json
import re
import logging
from pathlib import Path
from typing import Set, Dict, List, Optional, Any


logger = logging.getLogger(__name__)


class TagExtractor:
    """
    Extract tags from trajectory takeaways using LLM.
    
    Tags are categorized into:
    - Actions: #search, #filter, #click, #scroll, #type, #navigate
    - UI Elements: #search_bar, #dropdown, #button, #menu, #form
    - Strategies: #beginner, #free, #categories, #sort, #price
    """
    
    # Predefined tag categories for validation
    ACTION_TAGS = {
        '#search', '#filter', '#click', '#scroll', '#type', '#select',
        '#navigate', '#submit', '#enter', '#press', '#input', '#browse'
    }
    
    UI_ELEMENT_TAGS = {
        '#search_bar', '#dropdown', '#button', '#menu', '#link',
        '#input_field', '#form', '#tab', '#modal', '#checkbox',
        '#sidebar', '#header', '#footer', '#pagination'
    }
    
    STRATEGY_TAGS = {
        '#free', '#beginner', '#advanced', '#intermediate',
        '#categories', '#language', '#sort', '#price', '#rating',
        '#duration', '#certificate', '#university', '#provider'
    }
    
    ALL_KNOWN_TAGS = ACTION_TAGS | UI_ELEMENT_TAGS | STRATEGY_TAGS
    
    # Prompt template for LLM-based extraction
    EXTRACTION_PROMPT = """Extract 3-6 semantic tags from this GUI agent trajectory takeaway.

Tags should describe:
1. ACTIONS performed: #search, #filter, #click, #scroll, #type, #navigate
2. UI ELEMENTS used: #search_bar, #dropdown, #button, #menu, #form
3. STRATEGIES employed: #free, #beginner, #categories, #sort, #price

Takeaway: "{takeaway}"
Domain: {domain}

Return ONLY a JSON array of lowercase tags with # prefix. Example:
["#search", "#filter", "#beginner", "#search_bar"]

Tags:"""

    def __init__(self, llm=None, cache_path: Optional[str] = None):
        """
        Initialize the tag extractor.
        
        Args:
            llm: LLM instance for tag extraction (must have .chat() method)
            cache_path: Path to cache file for extracted tags
        """
        self.llm = llm
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, List[str]] = {}
        
        if self.cache_path and self.cache_path.exists():
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached tags from disk."""
        if self.cache_path is None:
            raise ValueError("cache_path must not be None when loading cache")
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Tag cache file not found: {self.cache_path}")

        data = json.loads(self.cache_path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Invalid tag cache format in {self.cache_path}: expected a JSON object")
        tags = data.get('tags')
        if not isinstance(tags, dict):
            raise ValueError(f"Invalid tag cache format in {self.cache_path}: missing/invalid 'tags' object")

        validated: Dict[str, List[str]] = {}
        for cache_key, tag_list in tags.items():
            if not isinstance(cache_key, str) or not cache_key:
                raise ValueError(f"Invalid tag cache key in {self.cache_path}: {cache_key!r}")
            if not isinstance(tag_list, list) or not tag_list:
                raise ValueError(f"Invalid tag cache entry for {cache_key!r} in {self.cache_path}: expected non-empty list")
            for t in tag_list:
                if not isinstance(t, str) or not t:
                    raise ValueError(f"Invalid cached tag for {cache_key!r} in {self.cache_path}: {t!r}")
            validated[cache_key] = tag_list

        self.cache = validated
        logger.info(f"[TagExtractor] Loaded {len(self.cache)} cached tag sets")
    
    def _save_cache(self) -> None:
        """Save cached tags to disk."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {'tags': self.cache}
            self.cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def extract_tags(self, takeaway: str, domain: str = "", 
                     trajectory_id: Optional[str] = None) -> Set[str]:
        """
        Extract tags from a takeaway.
        
        Args:
            takeaway: The trajectory summary text
            domain: Domain name (e.g., "Coursera")
            trajectory_id: Optional ID for caching
        
        Returns:
            Set of extracted tags
        """
        if not isinstance(takeaway, str) or not takeaway.strip():
            raise ValueError("takeaway must be a non-empty string")
        if not isinstance(domain, str):
            raise ValueError("domain must be a string")

        # Check cache first
        cache_key = trajectory_id or takeaway[:100]
        if cache_key in self.cache:
            logger.debug(f"[TagExtractor] Cache hit for {cache_key}")
            return set(self.cache[cache_key])
        
        # Extract tags
        if self.llm is None:
            raise ValueError("TagExtractor requires an LLM (llm must not be None)")
        
        tags = self._extract_with_llm(takeaway, domain)
        
        # NOTE: Domain tag removed to test pure semantic retrieval
        # Previously: tags.add(f"#{domain.lower()}")
        
        # Cache result
        self.cache[cache_key] = list(tags)
        self._save_cache()
        
        logger.info(f"[TagExtractor] Extracted tags for {cache_key}: {tags}")
        return tags
    
    def _extract_with_llm(self, takeaway: str, domain: str) -> Set[str]:
        """Extract tags using LLM."""
        prompt = self.EXTRACTION_PROMPT.format(
            takeaway=takeaway,
            domain=domain or "unknown"
        )
        
        messages = [
            {"role": "system", "content": "You are a tag extraction assistant. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response, _, _ = self.llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=100)
            
        if not hasattr(response, 'content'):
            raise ValueError("LLM response missing .content")
        response_text = response.content
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("LLM response content must be a non-empty string")
            
        # Parse JSON response
        tags = self._parse_llm_response(response_text)
        if not tags:
            raise ValueError(f"LLM returned no tags. response={response_text!r}")
        return tags
    
    def _parse_llm_response(self, response: str) -> Set[str]:
        """Parse LLM response to extract tags."""
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must be a non-empty string")

        match = re.search(r'\[[\s\S]*\]', response)
        if match is None:
            raise ValueError(f"No JSON array found in LLM response: {response!r}")

        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            raise ValueError(f"LLM response JSON must be a list, got {type(parsed)}")

        tags: Set[str] = set()
        for raw in parsed:
            if raw is None:
                raise ValueError("LLM returned null tag")
            tag = str(raw).lower().strip()
            if not tag:
                raise ValueError("LLM returned empty tag")
            if not tag.startswith('#'):
                tag = f'#{tag}'
            if re.fullmatch(r'#[a-z0-9_]+', tag) is None:
                raise ValueError(f"Invalid tag format: {tag!r}")
            tags.add(tag)
        
        if not tags:
            raise ValueError(f"Parsed zero tags from LLM response: {response!r}")
        
        return tags
    
    def _extract_with_rules(self, takeaway: str, domain: str) -> Set[str]:
        """
        Extract tags using rule-based matching.
        
        This helper is not used unless explicitly called.
        """
        tags = set()
        text = takeaway.lower()
        
        # Action keyword matching
        action_keywords = {
            'search': '#search',
            'filter': '#filter',
            'click': '#click',
            'scroll': '#scroll',
            'type': '#type',
            'select': '#select',
            'navigate': '#navigate',
            'submit': '#submit',
            'enter': '#enter',
            'press': '#press',
            'browse': '#browse',
        }
        
        for keyword, tag in action_keywords.items():
            if keyword in text:
                tags.add(tag)
        
        # UI element matching
        ui_keywords = {
            'search bar': '#search_bar',
            'searchbar': '#search_bar',
            'dropdown': '#dropdown',
            'button': '#button',
            'menu': '#menu',
            'link': '#link',
            'input': '#input_field',
            'form': '#form',
            'tab': '#tab',
            'modal': '#modal',
            'checkbox': '#checkbox',
            'sidebar': '#sidebar',
        }
        
        for keyword, tag in ui_keywords.items():
            if keyword in text:
                tags.add(tag)
        
        # Strategy matching
        strategy_keywords = {
            'free': '#free',
            'beginner': '#beginner',
            'advanced': '#advanced',
            'intermediate': '#intermediate',
            'categor': '#categories',  # matches category, categories
            'language': '#language',
            'sort': '#sort',
            'price': '#price',
            'rating': '#rating',
            'duration': '#duration',
            'certificate': '#certificate',
            'university': '#university',
            'provider': '#provider',
        }
        
        for keyword, tag in strategy_keywords.items():
            if keyword in text:
                tags.add(tag)
        
        return tags
    
    def extract_tags_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """
        Extract tags for multiple items.
        
        Args:
            items: List of dicts with 'id', 'takeaway', and optionally 'domain'
        
        Returns:
            Dict mapping trajectory_id to set of tags
        """
        results = {}
        for item in items:
            traj_id = item.get('id', '')
            takeaway = item.get('takeaway', '')
            domain = item.get('domain', '')
            
            if takeaway:
                tags = self.extract_tags(takeaway, domain, traj_id)
                results[traj_id] = tags
        
        return results
    
    def get_tag_statistics(self) -> Dict[str, int]:
        """Get statistics on tag usage from cache."""
        tag_counts: Dict[str, int] = {}
        
        for tags_list in self.cache.values():
            for tag in tags_list:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return dict(sorted(tag_counts.items(), key=lambda x: -x[1]))
