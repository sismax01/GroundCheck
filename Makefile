.PHONY: help install run test benchmark seed clean

PYTHON ?= python3

help:
	@echo "GroundCheck — Development & Evaluation Commands"
	@echo "  make install    Install dependencies via pip"
	@echo "  make run        Start the GroundCheck FastAPI server on port 8000"
	@echo "  make test       Run the automated test suite (unittest)"
	@echo "  make benchmark  Execute the reproducible evaluation benchmark via CLI"
	@echo "  make seed       Seed the database with the technical reference corpus"
	@echo "  make clean      Remove temporary build and cache artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m groundcheck.cli serve --host 127.0.0.1 --port 8000

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

benchmark:
	$(PYTHON) -m groundcheck.cli eval --mode hybrid --top-k 5 --alpha 0.55

seed:
	$(PYTHON) -m groundcheck.cli seed

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
