from subsets.learn import Modules

from .base import Generator


class Learn(Generator):
    def __init__(self, module, *args, **kwargs):
        if module not in Modules:
            raise KeyError(f"Learn module {module} is not registered")
        self.module = Modules[module](*args, **kwargs)

    def run(self, pool):
        self.module.init(system=pool.system)
        self.module.learn()
