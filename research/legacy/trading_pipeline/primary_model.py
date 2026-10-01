class PrimaryModel:
    """
    一次モデル: 価格の方向性およびバリア到達確率を予測する。
    LightGBM等の決定木ベースのモデルを利用し、イベント発生時のみ発火する。
    """
    def __init__(self):
        self.model = None

    def train(self, X, y):
        """
        LightGBM等を用いて学習を行う。
        """
        # TODO: LightGBMの学習ロジック実装
        pass

    def predict(self, X):
        """
        上側バリア、下側バリア、時間切れの各到達確率を予測。
        """
        # TODO: 推論ロジック実装
        pass
