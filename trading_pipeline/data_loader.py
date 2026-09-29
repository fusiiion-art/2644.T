import pandas as pd
from typing import Dict, List, Optional
import datetime

class DataLoader:
    """
    Point-in-Time (PiT) データ管理を伴うデータローダー。
    ルックアヘッドバイアス（未来情報の漏洩）を防ぐため、
    データがシステムで利用可能になった時刻 (available_time) を厳密に管理します。
    """
    def __init__(self):
        self.data_store: Dict[str, pd.DataFrame] = {}

    def load_historical_data(self, file_path: str, symbol: str) -> pd.DataFrame:
        """
        ヒストリカルデータを読み込む。
        実運用では、この関数内で 'known_at' や 'available_time' のカラムを生成・検証する。
        """
        # TODO: データソースに合わせた読み込み処理の実装
        pass

    def get_pit_data(self, target_time: datetime.datetime, symbols: List[str]) -> pd.DataFrame:
        """
        指定された時刻 (target_time) において、利用可能だった(known_at <= target_time)
        データのみを抽出して返す。これにより未来情報の混入を防ぐ。
        """
        # TODO: PiTに基づくデータ抽出ロジックの実装
        pass
