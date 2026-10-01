import re
import logging
from typing import List, Dict, Set

class FeatureRegistry:
    """
    特徴量セット定義（正規表現）に基づいて、指定されたカラムリストから
    使用する特徴量を動的に選択するクラス。
    """
    def __init__(self, requested_sets: List[str], definitions: Dict[str, List[str]]):
        if not requested_sets: 
            raise ValueError("No feature sets were requested.")
        self.requested_sets = requested_sets
        self.definitions = definitions
        self.feature_patterns: List[str] = self._get_feature_patterns()

    def _get_feature_patterns(self) -> List[str]:
        patterns = []
        for set_name in self.requested_sets:
            if set_name in self.definitions:
                set_def = self.definitions[set_name]
                if isinstance(set_def, str):
                    patterns.append(set_def)
                elif isinstance(set_def, list):
                    patterns.extend(set_def)
                else:
                    logging.warning(f"Feature set '{set_name}' has invalid type {type(set_def)} and will be skipped.")
            else:
                logging.warning(f"Undefined feature set '{set_name}' was requested and will be skipped.")
        return list(dict.fromkeys(patterns))

    def select_features(self, all_columns: List[str]) -> List[str]:
        selected_columns: Set[str] = set()
        for pattern in self.feature_patterns:
            try:
                regex = re.compile(pattern)
                matched = {col for col in all_columns if regex.search(col)}
                selected_columns.update(matched)
            except re.error as e:
                logging.error(f"Invalid regex pattern '{pattern}': {e}")
        
        # Hardcoded exclusion (legacy constraint, can be moved to config if needed)
        selected_columns.discard("maxis_allcountry_adj_close_return")
        
        logging.info(f"{len(selected_columns)} features selected. Example: {', '.join(sorted(list(selected_columns))[:5])}...")
        return sorted(list(selected_columns))
