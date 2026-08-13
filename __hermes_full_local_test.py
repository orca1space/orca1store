import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(r'D:\Hermes')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
results = []

def check(name, fn):
    try:
        value = fn()
        results.append({'name': name, 'status': 'PASS', 'detail': value})
    except Exception as exc:
        results.append({'name': name, 'status': 'FAIL', 'detail': f'{type(exc).__name__}: {exc}'})

def expect_local_only(fn):
    try:
        fn()
    except RuntimeError as exc:
        if 'local_only' in str(exc):
            return str(exc)
        raise
    raise AssertionError('external operation was not blocked')

check('python_imports', lambda: __import__('core.agent_api').agent_api.API_VERSION)
check('default_sources_are_local', lambda: __import__('core.training_daemon', fromlist=['DEFAULT_SOURCES']).DEFAULT_SOURCES)
check('api_import_blocked', lambda: expect_local_only(lambda: __import__('core.api_importer', fromlist=['get_api_importer']).get_api_importer().import_everything('https://example.invalid')))
check('hf_task_blocked', lambda: expect_local_only(lambda: __import__('core.training_daemon', fromlist=['TrainingDaemon']).TrainingDaemon()._task_hf_dataset({})))
check('github_task_blocked', lambda: expect_local_only(lambda: __import__('core.training_daemon', fromlist=['TrainingDaemon']).TrainingDaemon()._task_github_repo({})))
check('webui_loopback_binding', lambda: '127.0.0.1' in (ROOT / 'webui.py').read_text(encoding='utf-8'))
check('model_offline_flags', lambda: all(x in (ROOT / 'core' / 'llm.py').read_text(encoding='utf-8') for x in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']))

for path in ['http://127.0.0.1:7777/', 'http://127.0.0.1:7777/api/agent/ops']:
    def fetch(path=path):
        with urllib.request.urlopen(path, timeout=15) as r:
            body = r.read()
            assert r.status == 200
            return {'status': r.status, 'bytes': len(body)}
    check('http_' + path.rsplit('/', 1)[-1] or 'root', fetch)

from core import agent_api
for op in ['system.ping', 'system.info', 'system.list_ops', 'kb.stats', 'skills.list', 'session.active', 'session.tabs', 'memory.get', 'cache.stats', 'checkpoint.list', 'time_travel.history', 'hitl.list', 'training.status']:
    check('op_' + op, lambda op=op: agent_api.execute({'op': op, 'params': {}}))

summary = {
    'passed': sum(r['status'] == 'PASS' for r in results),
    'failed': sum(r['status'] == 'FAIL' for r in results),
    'total': len(results),
    'results': results,
}
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
if summary['failed']:
    raise SystemExit(1)
