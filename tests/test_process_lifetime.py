import asyncio
import ctypes
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class LifecycleTest(unittest.TestCase):
    def test_lifespan_signals_engine_and_awaits_batch_shutdown(self):
        from app import main
        calls = []
        class Engine:
            def load(self): calls.append('load')
            def start_resource_monitor(self): calls.append('start-monitor')
            def request_shutdown(self): calls.append('stop-engine')
            async def stop_resource_monitor(self): calls.append('stop-monitor')
        async def shutdown(): calls.append('stop-batches')
        async def run():
            async with main.lifespan(main.app):
                calls.append('serving')
        with patch.object(main, 'get_engine', return_value=Engine()), \
             patch.object(main, 'bind_process_lifetime'), \
             patch.object(main, 'shutdown_generation', side_effect=shutdown):
            asyncio.run(run())
        self.assertEqual(calls, ['load', 'start-monitor', 'serving', 'stop-engine', 'stop-monitor', 'stop-batches'])

    @unittest.skipUnless(os.name == 'nt', 'Windows job integration')
    def test_force_killing_parent_also_terminates_child(self):
        self._assert_tree_stops(False)

    @unittest.skipUnless(os.name == 'nt', 'Windows reload integration')
    def test_force_killing_reload_supervisor_stops_worker_and_child(self):
        self._assert_tree_stops(True)

    def _assert_tree_stops(self, reload_supervisor):
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL
        api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.TerminateProcess.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'child.pid'
            code = ('from app.process_lifetime import bind_process_lifetime; '
                    'import subprocess,sys,time; from pathlib import Path; '
                    'bind_process_lifetime(); '
                    'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                    'Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)')
            command = [sys.executable, '-c', code, str(marker)]
            if reload_supervisor:
                script = Path(directory) / 'reload_supervisor.py'
                script.write_text('''import multiprocessing, subprocess, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from app.process_lifetime import bind_process_lifetime
def worker(marker):
    bind_process_lifetime()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    Path(marker + '.worker').write_text(str(os.getpid()))
    Path(marker).write_text(str(child.pid))
    time.sleep(60)
if __name__ == '__main__':
    multiprocessing.get_context('spawn').Process(target=worker, args=(sys.argv[1],)).start()
    time.sleep(60)
''', encoding='utf-8')
                command = [sys.executable, str(script), str(marker)]
            parent = subprocess.Popen(command,
                                      cwd=Path(__file__).resolve().parents[1],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            child_handle = None
            worker_handle = None
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(marker.exists(), 'Parent failed to initialize job: ' +
                                (parent.stderr.read().decode() if parent.poll() is not None else 'timed out'))
                child_handle = api.OpenProcess(0x100001, False, int(marker.read_text()))  # SYNCHRONIZE | TERMINATE
                self.assertTrue(child_handle)
                if reload_supervisor:
                    worker_handle = api.OpenProcess(0x100001, False, int(Path(str(marker) + '.worker').read_text()))
                    self.assertTrue(worker_handle)
                self.assertEqual(api.WaitForSingleObject(child_handle, 0), 258)  # still running
                parent.kill()
                parent.wait(timeout=5)
                self.assertEqual(api.WaitForSingleObject(child_handle, 5000), 0)  # child exited
                if worker_handle:
                    self.assertEqual(api.WaitForSingleObject(worker_handle, 5000), 0)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=5)
                parent.stderr.close()
                for handle in (child_handle, worker_handle):
                    if handle:
                        if api.WaitForSingleObject(handle, 0) == 258:
                            api.TerminateProcess(handle, 1)
                            api.WaitForSingleObject(handle, 5000)
                        api.CloseHandle(handle)
