.PHONY: install test lint typecheck backtest run report cron-setup clean

install:
	python3 -m ensurepip --upgrade
	python3 -m pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ --max-line-length=100 --extend-ignore=E203,W503
	black --check src/

typecheck:
	mypy src/ --ignore-missing-imports

backtest:
	PYTHONPATH=. python3 src/backtesting/run_backtest.py

run:
	PYTHONPATH=. python3 -m scripts.trading.live_trader

report:
	PYTHONPATH=. python3 main.py report --testnet

cron-setup:
	@echo "Use cronjob tool to schedule hourly: PYTHONPATH=. python3 main.py trade --testnet"

init-db:
	PYTHONPATH=. python3 src/trading/init_db.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/*.db data/*.db-journal logs/*.log
