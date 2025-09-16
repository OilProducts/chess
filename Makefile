.PHONY: fmt lint typecheck test train

fmt:
	ruff --fix .
	black .

lint:
	ruff .

typecheck:
	mypy chessref

test:
	pytest

train:
	python -m chessref.train.train_supervised
