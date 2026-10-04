import base64
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from scripts import prepare_shared_cache as cache
from scripts.install_dependencies import install_command


class SharedCacheTest(unittest.TestCase):
    def test_installed_export_skips_relocated_wrappers_but_rejects_missing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); output = base / 'wheels'; output.mkdir()
            files = {'sample/__init__.py': b'value=1',
                     'sample-1.0.dist-info/METADATA': b'Name: sample\nVersion: 1.0\n',
                     'sample-1.0.dist-info/WHEEL': b'Root-Is-Purelib: true\nTag: py3-none-any\n'}
            for name, data in files.items():
                path = base / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            entries = list(files) + ['relocated/Scripts/sample.exe']
            distribution = types.SimpleNamespace(files=entries,
                entry_points=[types.SimpleNamespace(name='sample', group='console_scripts')],
                locate_file=lambda path: base / path,
                read_text=lambda name: files[f'sample-1.0.dist-info/{name}'].decode())
            self.assertTrue(cache.export_installed_portable(distribution, output))
            target = output / 'sample-1.0-py3-none-any.whl'
            with zipfile.ZipFile(target) as archive:
                self.assertNotIn('relocated/Scripts/sample.exe', archive.namelist())
            target.unlink()
            entries.append('sample/missing.py')
            self.assertFalse(cache.export_installed_portable(distribution, output))
            self.assertFalse(target.exists())

    def wheel(self, path, tag='py3-none-any', native=False):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('sample-1.0.dist-info/WHEEL', f'Root-Is-Purelib: true\nTag: {tag}\n')
            archive.writestr('sample/data.dll' if native else 'sample/__init__.py', b'data')

    def test_only_portable_archives_are_copied_and_repeated_runs_reuse_them(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'pip'; source.mkdir()
            destination = Path(directory) / 'common'; destination.mkdir()
            self.wheel(source / 'portable.body')
            self.wheel(source / 'windows.body', 'cp311-cp311-win_amd64')
            self.wheel(source / 'mislabelled.body', native=True)
            (source / 'not-a-wheel.body').write_text('metadata')
            self.assertEqual(cache.copy_cached_wheels(source, destination), 1)
            self.assertEqual(cache.copy_cached_wheels(source, destination), 0)
            self.assertEqual([p.name for p in destination.iterdir()], ['sample-1.0-py3-none-any.whl'])

    def test_model_export_keeps_compatibility_and_valid_record_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'site'; base.mkdir()
            output = Path(directory) / 'wheels'; output.mkdir()
            files = {'en_core_web_sm/meta.json': b'{"spacy_version":">=3.8.0,<3.9.0"}',
                     'en_core_web_sm/__init__.py': b'__version__="3.8.0"',
                     'en_core_web_sm-3.8.0.dist-info/METADATA': b'Metadata-Version: 2.1\nName: en-core-web-sm\nVersion: 3.8.0\n',
                     'en_core_web_sm-3.8.0.dist-info/WHEEL': b'Root-Is-Purelib: true\nTag: py3-none-any\n',
                     'en_core_web_sm-3.8.0.dist-info/direct_url.json': b'omit installation provenance'}
            for name, data in files.items():
                path = base / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            distribution = types.SimpleNamespace(version='3.8.0', files=list(files),
                locate_file=lambda path: base / path, read_text=lambda name: files[f'en_core_web_sm-3.8.0.dist-info/{name}'].decode())
            with patch.object(cache.metadata, 'distribution', return_value=distribution):
                wheel = cache.export_spacy_model(output)
                self.assertEqual(cache.export_spacy_model(output), wheel)
            with zipfile.ZipFile(wheel) as archive:
                self.assertNotIn('en_core_web_sm-3.8.0.dist-info/direct_url.json', archive.namelist())
                self.assertIn(b'Requires-Dist: spacy>=3.8.0,<3.9.0', archive.read('en_core_web_sm-3.8.0.dist-info/METADATA'))
                records = archive.read('en_core_web_sm-3.8.0.dist-info/RECORD').decode()
                for name, digest, size in csv.reader(io.StringIO(records)):
                    if digest:
                        data = archive.read(name)
                        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                        self.assertEqual(digest, 'sha256=' + actual)
                        self.assertEqual(int(size), len(data))

    def test_installer_selects_one_newest_model_and_resolves_it_with_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in ['3.8.0', '3.10.0']:
                (root / f'en_core_web_sm-{version}-py3-none-any.whl').touch()
            command, found = install_command(Path('requirements.txt'), root)
            self.assertTrue(found)
            self.assertTrue(command[-1].endswith('3.10.0-py3-none-any.whl'))
            self.assertIn('--find-links', command)
            self.assertIn('-r', command)

    def test_pinned_portable_packages_use_explicit_local_paths_and_keep_extras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / 'requirements.txt'
            requirements.write_text('uvicorn[standard]==0.35.0\nnumpy>=2\n')
            (root / 'uvicorn-0.35.0-py3-none-any.whl').touch()
            command, _ = install_command(requirements, root)
            self.assertTrue(any(arg.startswith('uvicorn[standard] @ file:') for arg in command))
