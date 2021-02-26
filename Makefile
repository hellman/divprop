PYTHON = pypy3

lib:
	poetry build 
	$(PYTHON) -m pip install -U .
	sage -pip install .
	make test

test:
	$(PYTHON) -m pytest tests/
	sage -sh -c 'pytest tests/ tests_sage/'

clean:
	rm -rf build setup.py *.egg-info __pycache__
	rm -f src/divprop/subsets/*_wrap*
	rm -f src/divprop/*_wrap*
	rm -f src/divprop/libsubsets.py