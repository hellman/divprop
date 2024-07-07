test:
	python -m pytest --doctest-modules tests/ src/
	python -m pytest --doctest-modules tests_sage/
	#sage -sh -c 'pytest tests/ tests_sage/'

clean:
	rm -rf build
	rm -f src/divprop/*_wrap.cpp
	rm -f src/divprop/lib.py
	rm -f src/divprop/*.so

venv:
	sage -python -m venv --system-site-packages .envsage/
	echo `pwd`/src >.envsage/lib/python*/site-packages/divprop.pth
	ln -sf .envsage/bin/activate ./activate

upload:
	rm -f dist/*
	python -m setuptools_scm
	python -m build
	echo
	echo "Upload version? or Ctrl+C"
	python -m setuptools_scm
	read confirm
	twine upload --repository divprop  dist/divprop-*.tar.gz

scm:
	python -m setuptools_scm

dev:
	pip install --no-build-isolation -e .