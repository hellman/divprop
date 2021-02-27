PYTHON = python

quick:
	# python setup.py build
	bash -c 'python setup.py install -f &>/dev/null'

lib:
	rm -f setup.py
	poetry build 
	$(PYTHON) -m pip install -U .
	sage -pip install .
	python .create_setup.py
	make test

test:
	$(PYTHON) -m pytest tests/
	$(PYTHON) -m pytest tests/ tests_sage/
	#sage -sh -c 'pytest tests/ tests_sage/'

clean:
	rm -rf build setup.py *.egg-info __pycache__
	rm -f src/divprop/subsets/*_wrap*
	rm -f src/divprop/*_wrap*
	rm -f src/divprop/libsubsets.py
	rm -f src/divprop/*.so

venv:
	sage -python -m venv --system-site-packages .envsage/
