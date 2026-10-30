"""Read-only source and documentation checks; never imports hardware scripts."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    counts = {'python': 0, 'notebooks': 0, 'links': 0}
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p.startswith('.') for p in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix == '.py':
            counts['python'] += 1
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=rel)
            except SyntaxError as exc:
                errors.append(f'{rel}:{exc.lineno}: {exc.msg}')
        elif path.suffix == '.ipynb':
            counts['notebooks'] += 1
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            for index, cell in enumerate(data['cells']):
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                # Ignore IPython command lines; check the surrounding Python.
                source = '\n'.join('pass' if line.lstrip().startswith(('!', '%')) else line for line in source.splitlines())
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    errors.append(f'{rel}:cell {index + 1}:line {exc.lineno}: {exc.msg}')
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'{rel}:cell {index + 1}: publication outputs not cleared')
        elif path.suffix == '.md':
            # Archived source notes retain historical links and are explicitly superseded.
            if '/archive/' in rel or '/legacy/' in rel or path.name == 'original-phase4-notes.md':
                continue
            text = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
            for link in re.findall(r'\]\(([^)]+)\)', text):
                target = link.split('#', 1)[0].strip('<>')
                if not target or re.match(r'\w+://', target):
                    continue
                counts['links'] += 1
                if not (path.parent / unquote(target)).exists():
                    errors.append(f'{rel}: missing target {target}')
    print(json.dumps(counts))
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
