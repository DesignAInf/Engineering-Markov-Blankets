.PHONY: install test quick reproduce paper clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

quick:
	python -m emb.experiments --quick --out runs/quick

reproduce:
	python -m emb.experiments --out runs/full

paper:
	cd paper && pdflatex engineering_markov_blankets_v9.tex && pdflatex engineering_markov_blankets_v9.tex

clean:
	rm -rf runs build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
