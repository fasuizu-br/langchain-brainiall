.PHONY: test test-unit test-integration lint format type-check build clean

test: test-unit

test-unit:
	python -m pytest tests/unit_tests -v

test-integration:
	python -m pytest tests/integration_tests -v -m integration

lint:
	python -m ruff check langchain_brainiall/ tests/

format:
	python -m ruff format langchain_brainiall/ tests/

type-check:
	python -m mypy langchain_brainiall/

build: clean
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info langchain_brainiall/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository testpypi dist/*
