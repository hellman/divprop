lib:
	poetry build && python3 -m pip install -U .
	pytest tests/

clean:
	rm -rf build *.egg-info __pycache__ src/subsets/__pycache__
	rm -f src/subsets/*_wrap*
	rm -f src/subsets/lib.py