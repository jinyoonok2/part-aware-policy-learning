"""Freeze the reusable packages alongside a prepared dataset or expert run."""
from pathlib import Path
import hashlib
import importlib.util
import shutil
import sys


def load_source_module(name, path):
    """Load immutable extractors, including their historical flat helper import."""
    from data_preparation import data_evaluation_reservations
    previous = sys.modules.get('data_evaluation_reservations')
    if previous is None:
        sys.modules['data_evaluation_reservations'] = data_evaluation_reservations
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop('data_evaluation_reservations', None)


def source_hashes(source):
    source = Path(source)
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts}


def freeze_packages(project, source):
    """Copy executable dependencies, keeping package imports valid in snapshots.

    Historical flat entry points remain available for existing Slurm commands.
    New package copies make those entry points independent of the live project.
    """
    project, source = Path(project), Path(source)
    for name in ('data_preparation', 'grasp'):
        package = project / name
        if not package.is_dir():
            raise FileNotFoundError(f'Missing required package: {package}')
        for path in sorted(package.rglob('*')):
            if not path.is_file() or path.suffix not in ('.py', '.sh', '.json'):
                continue
            if '__pycache__' in path.parts:
                continue
            target = source / name / path.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
    return source_hashes(source)
