"""Monte Carlo Tree Search guidance for iterative refiner."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import torch

from chessref.data.planes import board_to_planes
from chessref.inference.predict import refine_predict
from chessref.model.refiner import IterativeRefiner
from chessref.moves.encoding import encode_move
from chessref.moves.legal_mask import legal_move_mask


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    inference_max_loops: int = 1


class _Node:
    __slots__ = ("parent", "move", "prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float, move: Optional[chess.Move] = None, parent: Optional["_Node"] = None) -> None:
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[chess.Move, _Node] = {}

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, priors: Dict[chess.Move, float]) -> None:
        for move, prior in priors.items():
            if move not in self.children:
                self.children[move] = _Node(prior=prior, move=move, parent=self)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self, c_puct: float) -> "_Node":
        total_visits = max(1, sum(child.visit_count for child in self.children.values()))
        best_score = -float("inf")
        best_child = None
        for child in self.children.values():
            u = c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            score = child.q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child


class MCTS:
    def __init__(self, model: IterativeRefiner, config: Optional[MCTSConfig] = None, *, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.cfg = config or MCTSConfig()
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def search(self, board: chess.Board, *, add_noise: bool = True) -> Dict[chess.Move, int]:
        root = _Node(prior=1.0)

        policy, value = self._evaluate(board)
        root.expand(policy)
        root.visit_count = 1
        root.value_sum = value

        if add_noise and self.cfg.dirichlet_alpha > 0 and root.children:
            noise = torch.distributions.Dirichlet(
                torch.full((len(root.children),), self.cfg.dirichlet_alpha)
            ).sample().tolist()
            for noise_val, child in zip(noise, root.children.values()):
                child.prior = child.prior * (1 - self.cfg.dirichlet_epsilon) + noise_val * self.cfg.dirichlet_epsilon

        for _ in range(self.cfg.num_simulations):
            board_copy = board.copy(stack=False)
            node = root
            path = [node]

            # Selection
            while node.is_expanded() and node.children:
                node = node.select_child(self.cfg.c_puct)
                board_copy.push(node.move)
                path.append(node)

            leaf_value = self._expand_leaf(node, board_copy)

            # Backpropagate
            value_to_propagate = leaf_value
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value_to_propagate
                value_to_propagate = -value_to_propagate

        visit_counts: Dict[chess.Move, int] = {move: child.visit_count for move, child in root.children.items()}
        return visit_counts

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> (Dict[chess.Move, float], float):
        planes = board_to_planes(board).unsqueeze(0).to(self.device)
        legal_mask = legal_move_mask(board).unsqueeze(0).to(self.device)
        result = refine_predict(
            self.model,
            planes,
            legal_mask,
            max_loops=self.cfg.inference_max_loops,
            min_loops=1,
            use_act=self.model.use_act if hasattr(self.model, "use_act") else False,  # type: ignore[attr-defined]
        )
        policy = result.final_policy[0]
        value = result.final_value[0].item()

        move_priors: Dict[chess.Move, float] = {}
        for move in board.legal_moves:
            idx = encode_move(move, board)
            move_priors[move] = float(policy[idx])

        total = sum(move_priors.values())
        if total > 0:
            for move in move_priors:
                move_priors[move] /= total
        else:  # fallback to uniform
            uniform = 1.0 / max(1, len(move_priors))
            for move in move_priors:
                move_priors[move] = uniform

        return move_priors, value

    @torch.no_grad()
    def _expand_leaf(self, node: _Node, board: chess.Board) -> float:
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            if result == "1-0":
                return 1.0
            if result == "0-1":
                return -1.0
            return 0.0

        priors, value = self._evaluate(board)
        node.expand(priors)
        return value


__all__ = ["MCTS", "MCTSConfig"]
