.PHONY: install infra-up infra-down run test lint

install:
	python -m pip install -e ".[dev]"

infra-up:
	docker compose up -d

infra-down:
	docker compose down

run:
	streamlit run app.py

test:
	pytest

lint:
	ruff check .

