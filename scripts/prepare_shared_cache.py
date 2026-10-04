"""Export reusable downloads without accessing the network.

python -m scripts.prepare_shared_cache
"""
import argparse
import base64
import csv
from email.parser import Parser
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import shutil
import zipfile
import re

ROOT = Path(__file__).resolve().parent.parent


def dependency_distributions():
    pending = ['fastapi', 'uvicorn', 'pydantic-settings', 'kokoro', 'soundfile',
               'numpy', 'python-multipart', 'psutil', 'en-core-web-sm']
    seen = set()
    while pending:
        name = pending.pop().lower().replace('_', '-').replace('.', '-')
        if name in seen:
            continue
        seen.add(name)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        yield distribution
        for requirement in distribution.requires or []:
            match = re.match(r'[A-Za-z0-9][A-Za-z0-9._-]*', requirement)
            if match:
                pending.append(match.group())


def export_installed_portable(distribution, destination):
    """Rebuild pure wheels only when all payload paths are recoverable safely."""
    wheel = Parser().parsestr(distribution.read_text('WHEEL') or '')
    tags = wheel.get_all('Tag', [])
    entries = distribution.files or []
    if wheel.get('Root-Is-Purelib', '').lower() != 'true' or not tags or any(
            not tag.endswith('-none-any') for tag in tags):
        return False
    if any(str(path).lower().endswith(('.dll', '.so', '.pyd', '.dylib', '.pth')) for path in entries):
        return False
    dist_info = next((str(path).split('/')[0] for path in entries
                      if str(path).endswith('.dist-info/METADATA')), None)
    if not dist_info:
        return False
    target = destination / f'{dist_info.removesuffix(".dist-info")}-{tags[-1]}.whl'
    if target.exists():
        return False
    base = Path(distribution.locate_file('')).resolve()
    wrappers = set()
    for entry in distribution.entry_points:
        if entry.group in {'console_scripts', 'gui_scripts'}:
            wrappers.update({entry.name, entry.name + '.exe', entry.name + '-script.py'})
    files = {}
    for entry in entries:
        relative = Path(entry)
        if '__pycache__' in relative.parts or relative.suffix == '.pyc':
            continue
        source = Path(distribution.locate_file(entry)).resolve()
        if not source.is_relative_to(base):
            # pip regenerates these wrappers from entry_points.txt on Linux.
            if relative.name in wrappers:
                continue
            return False  # Cannot reconstruct arbitrary .data relocation.
        if relative.name in {'RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json'} and dist_info in relative.parts:
            continue
        if not source.is_file():
            if relative.name in wrappers:
                continue  # Some Conda RECORD files contain relocated wrapper paths.
            return False
        files[relative.as_posix()] = source.read_bytes()
    records = io.StringIO(newline='')
    writer = csv.writer(records, lineterminator='\n')
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, len(data)])
    record = f'{dist_info}/RECORD'
    writer.writerow([record, '', ''])
    files[record] = records.getvalue().encode()
    temporary = target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    temporary.replace(target)
    return True


def portable_wheel_name(archive):
    wheels = [name for name in archive.namelist() if name.endswith('.dist-info/WHEEL')]
    if len(wheels) != 1:
        return None
    wheel = Parser().parsestr(archive.read(wheels[0]).decode())
    tags = wheel.get_all('Tag', [])
    if wheel.get('Root-Is-Purelib', '').lower() != 'true' or not tags or any(
            not tag.endswith('-none-any') for tag in tags):
        return None
    if any(name.lower().endswith(('.dll', '.pyd', '.so', '.dylib')) for name in archive.namelist()):
        return None
    prefix = wheels[0].split('/')[0].removesuffix('.dist-info')
    return f'{prefix}-{tags[-1]}.whl'


def copy_cached_wheels(cache, destination):
    count = 0
    if not cache.is_dir():
        return count
    for path in cache.rglob('*'):
        if not path.is_file() or path.suffix not in {'.whl', '.body'}:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                name = portable_wheel_name(archive)
            if name:
                target = destination / name
                if not target.exists():
                    temporary = target.with_suffix('.tmp')
                    shutil.copyfile(path, temporary)
                    temporary.replace(target)
                    count += 1
        except (OSError, ValueError, zipfile.BadZipFile, UnicodeError):
            continue
    return count


def export_spacy_model(destination):
    """Reconstruct only the known portable model-data wheel, never site-packages."""
    try:
        distribution = metadata.distribution('en-core-web-sm')
    except metadata.PackageNotFoundError:
        return None
    wheel = Parser().parsestr(distribution.read_text('WHEEL') or '')
    if wheel.get('Root-Is-Purelib', '').lower() != 'true' or wheel.get_all('Tag') != ['py3-none-any']:
        raise ValueError('The installed English model is not a portable data wheel')
    dist_info = next(str(path).split('/')[0] for path in distribution.files
                     if str(path).endswith('.dist-info/METADATA'))
    name = f'en_core_web_sm-{distribution.version}-py3-none-any.whl'
    target = destination / name
    if target.exists():
        return target
    base = Path(distribution.locate_file('')).resolve()
    files = {}
    for entry in distribution.files:
        relative = Path(entry)
        if relative.parts[0] not in {'en_core_web_sm', dist_info} or '__pycache__' in relative.parts:
            continue
        if relative.name in {'RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json'} or relative.suffix == '.pyc':
            continue
        source = Path(distribution.locate_file(entry)).resolve()
        if not source.is_relative_to(base):
            raise ValueError('Model resource escapes the Python environment')
        files[relative.as_posix()] = source.read_bytes()
    # spaCy data packages sometimes omit Requires-Dist; retain their actual
    # compatibility range so the container resolves a matching spaCy version.
    model_meta = json.loads(files['en_core_web_sm/meta.json'])
    info = Parser().parsestr(files[f'{dist_info}/METADATA'].decode())
    if not any(requirement.lower().startswith('spacy') for requirement in info.get_all('Requires-Dist', [])):
        info['Requires-Dist'] = 'spacy' + model_meta['spacy_version']
        files[f'{dist_info}/METADATA'] = info.as_string().encode()
    records = io.StringIO(newline='')
    writer = csv.writer(records, lineterminator='\n')
    for relative, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([relative, 'sha256=' + digest, len(data)])
    record = f'{dist_info}/RECORD'
    writer.writerow([record, '', ''])
    files[record] = records.getvalue().encode()
    temporary = target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, data in files.items():
            archive.writestr(relative, data)
    temporary.replace(target)
    return target


def prepare(destination, caches):
    destination.mkdir(parents=True, exist_ok=True)
    # Export the installed language model first so its compatibility metadata
    # takes precedence over a cached wheel of the same version.
    model = export_spacy_model(destination)
    installed = sum(export_installed_portable(distribution, destination)
                    for distribution in dependency_distributions())
    count = sum(copy_cached_wheels(cache, destination) for cache in set(caches))
    return {'newInstalledWheels': installed, 'newCachedWheels': count, 'languageModel': model.name if model else None,
            'portableWheelCount': len(list(destination.glob('*.whl')))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/wheels/common')
    parser.add_argument('--pip-cache', type=Path, action='append', default=[])
    args = parser.parse_args()
    caches = args.pip_cache or [ROOT / 'data/cache/pip/nt', ROOT / 'data/cache/pip/posix']
    if os.environ.get('PIP_CACHE_DIR'):
        caches.append(Path(os.environ['PIP_CACHE_DIR']))
    if os.name == 'nt' and os.environ.get('LOCALAPPDATA'):
        caches.append(Path(os.environ['LOCALAPPDATA']) / 'pip/Cache')
    elif os.name != 'nt':
        caches.append(Path.home() / '.cache/pip')
    print(json.dumps(prepare(args.destination, caches), indent=2))
