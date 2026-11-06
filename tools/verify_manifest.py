from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parents[1]
m=json.loads((r/'MANIFEST.json').read_text())
for e in m['release_files']:
 p=r/e['path']
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],e['path']
print('Verified',len(m['release_files']),'release files')
