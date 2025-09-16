from chessref.eval.eval_match import MatchConfig, MatchResult, evaluate_matches
from chessref.train.train_supervised import ModelConfig


def test_evaluate_matches_random_opponent() -> None:
    model_cfg = ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False)
    cfg = MatchConfig(
        model=model_cfg,
        checkpoint=None,
        num_games=1,
        max_moves=20,
        opponent="random",
        device="cpu",
        inference_max_loops=1,
    )

    result = evaluate_matches(cfg)
    assert isinstance(result, MatchResult)
    assert result.wins + result.losses + result.draws == 1
