quick:
	#python setup.py build
	#bash -c 'python setup.py install -f &>/dev/null'
	bash -c 'python setup.py bdist_wheel &>/dev/null'
	python -m pip install --upgrade --no-deps --force-reinstall dist/divprop-*-cp38-cp38-linux_x86_64.whl

quickpypy:
	pypy3 setup.py bdist_wheel
	pypy3 -m pip install --upgrade --no-deps --force-reinstall dist/divprop-*-pp37-pypy37_pp73-linux_x86_64.whl

lib:
	python setup.py bdist_wheel
	python -m pip install -U .
	#make test

test:
	python -m pytest tests/
	python -m pytest tests/ tests_sage/
	#sage -sh -c 'pytest tests/ tests_sage/'

clean:
	rm -rf build setup.py *.egg-info __pycache__
	rm -f src/divprop/subsets/*_wrap*
	rm -f src/divprop/*_wrap*
	rm -f src/divprop/libsubsets.py
	rm -f src/divprop/*.so

venv:
	sage -python -m venv --system-site-packages .envsage/
