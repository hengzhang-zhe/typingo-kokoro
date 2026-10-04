"""Resolve dependencies and a shared language-model wheel in one pip invocation."""
import argparse
from pathlib import Path
import subprocess
import sys
import re


def install_command(requirements, wheelhouse):
    command = [sys.executable, '-m', 'pip', 'install', '--find-links', str(wheelhouse),
               '-r', str(requirements)]
    # pip gives index URLs and find-links equal priority. Explicit paths ensure
    # pinned portable project packages reuse local bytes even on a first build.
    if requirements.is_file():
        for line in requirements.read_text().splitlines():
            match = re.fullmatch(r'([\w.-]+)(?:\[([^]]+)\])?==([\w.+-]+)', line.strip())
            if not match:
                continue
            name, extras, version = match.groups()
            canonical = re.sub(r'[-_.]+', '_', name).lower()
            candidates = sorted(wheelhouse.glob(f'{canonical}-{version}-*-none-any.whl'))
            if candidates:
                path = candidates[-1].resolve().as_uri()
                command.append(f'{name}[{extras}] @ {path}' if extras else f'{name} @ {path}')
    models = [path for path in wheelhouse.glob('en_core_web_sm-*-py3-none-any.whl')
              if re.fullmatch(r'en_core_web_sm-\d+\.\d+\.\d+-py3-none-any\.whl', path.name)]
    if models:
        latest = max(models, key=lambda p: tuple(map(int, p.name.split('-')[1].split('.'))))
        command.append(str(latest))
    return command, bool(models)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--requirements', type=Path, required=True)
    args = parser.parse_args()
    command, has_model = install_command(args.requirements, args.wheelhouse)
    subprocess.run(command, check=True)
    if not has_model:
        # First-time Docker-only installs fetch once into the persistent pip
        # build cache. Subsequent builds reuse the layer/cache.
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
