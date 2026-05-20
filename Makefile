.PHONY: install install-dev test lint format type-check clean docker

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"

test:
	pytest -v

test-cov:
	pytest --cov=hscode --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests

format:
	ruff format src tests

type-check:
	mypy src

preload-cache:
	hscode --preload-cache

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

docker:
	docker build -t hscode .
