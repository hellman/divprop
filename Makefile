lib:
	poetry build && python3 -m pip install -U .
	make test

test:
	pytest tests/

clean:
	rm -rf build setup.py *.egg-info __pycache__
	rm -f src/divprop/subsets/*_wrap*
	rm -f src/divprop/*_wrap*
	rm -f src/divprop/libsubsets.py