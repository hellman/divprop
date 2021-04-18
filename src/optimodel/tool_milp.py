import os
import sys

import argparse
from argparse import RawTextHelpFormatter

from subsets import DenseSet, SparseSet
from subsets.learn import Modules as LearnModules

from optimodel.pool import InequalitiesPool, TypeGood

import divprop.logs as logging

# sage/pure python compatibility
try:
    import sage.all
except ImportError:
    pass


AutoSimple = (
    "Learn:LevelLearn,levels_lower=3",
    # "Learn:RandomLower:max_repeat_rate=3",
    "Learn:GainanovSAT,sense=min,save_rate=100,solver=cadical",
    # min vs None?
    "SubsetGreedy:",
    "SubsetWriteMILP:",
    "SubsetMILP:",
)
AutoShifts = (
    "Sub:Learn:LevelLearn,levels_lower=3",
    "Sub:Learn:GainanovSAT,sense=min,save_rate=100,solver=cadical"
    "ShiftLearn:threads=4",
)


class ToolMILP:
    log = logging.getLogger(__name__)

    def main(self):
        TOOL = os.path.basename(sys.argv[0])

        logging.setup(level="INFO")

        parser = argparse.ArgumentParser(description=f"""
    Generate inequalities to model a set.
    AutoSimple: alias for
        {" ".join(AutoSimple)}
    AutoShifts: alias for
        {" ".join(AutoShifts)}
        """.strip(), formatter_class=RawTextHelpFormatter)

        parser.add_argument(
            "fileprefix", type=str,
            help="Sets prefix "
            "(files with appended .good.set, .bad.set, .type_good must exist)",
        )
        parser.add_argument(
            "commands", type=str, nargs="*",
            help="Commands with options (available: Learn:* ???)",
        )

        args = self.args = parser.parse_args()

        self.fileprefix = args.fileprefix

        assert os.path.exists(self.fileprefix + ".good.set")
        assert os.path.exists(self.fileprefix + ".bad.set")
        assert os.path.exists(self.fileprefix + ".type_good")

        logging.addFileHandler(self.fileprefix + f".log.{TOOL}")

        self.log.info(args)

        self.pool = InequalitiesPool.from_DenseSet_files(
            fileprefix=self.fileprefix,
        )

        commands = args.commands
        if self.pool.is_monotone:
            commands = commands or AutoSimple
        else:
            commands = commands or AutoShifts

        self.log.info(f"commands: {' '.join(commands)}")

        self.output_ineqs = args.fileprefix + ".ineqs"
        self.log.info(f"using output prefix {self.output_ineqs}")

        for cmd in commands:
            self.run_command_string(cmd)

    def run_command_string(self, cmd):
        method, args, kwargs = parse_method(cmd)
        self.log.info(f"running command {method} {args} {kwargs}")
        ret = getattr(self, method)(*args, **kwargs)
        self.log.info(f"command {method} returned {ret}")

    def AutoSimple(self):
        for cmd in AutoSimple:
            self.run_command_string(cmd)

    def AutoShifts(self):
        for cmd in AutoShifts:
            self.run_command_string(cmd)

    def Learn(self, module, *args, **kwargs):
        if module not in LearnModules:
            raise KeyError(f"Learn module {module} is not registered")
        self.module = LearnModules[module](*args, **kwargs)
        self.module.init(system=self.pool.system)
        self.module.learn()

    def ShiftLearn(self):
        pass

    def SubsetGreedy(self, *args, **kwargs):
        res = self.pool.choose_subset_greedy(*args, **kwargs)
        self.save_ineqs(res)

    def SubsetMILP(self, *args, **kwargs):
        res = self.pool.choose_subset_milp(*args, **kwargs)
        self.save_ineqs(res)

    def SubsetWriteMILP(self, *args, **kwargs):
        prefix = os.path.join(self.fileprefix, "lp/")
        os.makedirs(prefix, exist_ok=True)
        prefix = os.path.join(prefix, "lp/full")

        self.pool.write_subset_milp(filename=prefix + ".lp", **kwargs)

    #     "ShiftLearn": NotImplemented,
    #     "Polyhedron": NotImplemented,

    def save_ineqs(self, ineqs):
        filename = f"{self.output_ineqs}.{len(ineqs)}"

        if os.path.exists(filename):
            self.log.warning(f"file {filename} exists, skipping overwrite!")
        else:
            self.log.info(f"saving {len(ineqs)} ineqs to {filename}")
            with open(filename, "w") as f:
                print(len(ineqs), file=f)
                for eq in ineqs:
                    print(*eq, file=f)
            self.log.info(f"saved {len(ineqs)} ineqs to {filename}")

        if len(ineqs) < 50:
            self.log.info(f"inequalities ({len(ineqs)}):")
            for ineq in ineqs:
                self.log.info(f"{ineq}")
            self.log.info("end")


# def separate_monotonic(
#         points_good, points_bad, type_good, gens, subset_method=DEFAULT_SUBSET,
#     ):
#     assert type_good in ("lower", "upper", None)
#     subset_method, subset_args, subset_kwargs = parse_method(subset_method)
#     assert subset_method in ("milp", "greedy")

#     pool = InequalitiesPool(
#         points_good=points_good,
#         points_bad=points_bad,
#         type_good=type_good,
#     )

#     GENERATORS = {
#         "gemcut": pool.generate_GemCut,
#         "randomgroupcut": pool.generate_RandomGroupCut,
#         "randomplanecut": pool.generate_RandomPlaneCut,
#         "polyhedron": pool.generate_from_polyhedron,
#     }
#     for gen in gens:
#         gen_name, gen_args, gen_kwargs = parse_method(gen)
#         GENERATORS[gen_name.lower()](*gen_args, **gen_kwargs)

#     pool.log_stat()

#     pool.check()

#     METHODS = {
#         "greedy": pool.choose_subset_greedy,
#         "milp": pool.choose_subset_milp,
#     }
#     Lstar = METHODS[subset_method](*subset_args, **subset_kwargs)

#     pool.log_stat(Lstar)

#     pool.check(Lstar)

#     return Lstar


def parse_method(s):
    """
    >>> parse_method("Test")
    ('Test', (), {})
    >>> parse_method("Test:")
    ('Test', (), {})
    >>> parse_method("Test:asd")
    ('Test', ('asd',), {})
    >>> parse_method("Test:test,asd=123")
    ('Test', ('test',), {'asd': 123})
    >>> parse_method("Test:asd=123a")
    ('Test', (), {'asd': '123a'})
    >>> parse_method("Pre:Test,asd")
    ('Pre', ('Test', 'asd'), {})
    """
    if ":" not in s:
        s += ":"
    method, str_opts = s.split(":", 1)
    assert method

    kwargs = {}
    args = []
    str_opts = str_opts.strip()
    if str_opts:
        for opt in str_opts.split(","):
            if "=" in opt:
                key, val = opt.split("=", 1)
                kwargs[key] = parse_value(val)
            else:
                val = opt
                args.append(parse_value(val))
    return method, tuple(args), kwargs


def parse_value(s: str):
    """
    >>> parse_value("123")
    123
    >>> parse_value("123.0")
    123.0
    >>> parse_value("+inf")
    inf
    >>> parse_value("123a")
    '123a'
    >>> parse_value("None")
    >>> parse_value("False")
    False
    >>> parse_value("True")
    True
    >>> parse_value("true")
    'true'
    """
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s == "None":
        return
    if s == "False":
        return False
    if s == "True":
        return True
    return s


def main():
    return ToolMILP().main()


if __name__ == '__main__':
    main()
