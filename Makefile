.PHONY: test demo ui api notebook

test:
	.venv/bin/pytest -q

demo:
	.venv/bin/python scripts/demo.py

ui:
	.venv/bin/streamlit run demo/app.py

api:
	.venv/bin/uvicorn src.serving:app --port 8000

notebook:
	.venv/bin/jupyter notebook notebooks/01_eda_bandit.ipynb
