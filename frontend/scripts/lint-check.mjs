import fs from 'node:fs';
import path from 'node:path';

const SRC_DIR = path.resolve('src');

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else if (/\.(js|jsx)$/.test(entry.name)) out.push(full);
  }
  return out;
}

const files = walk(SRC_DIR);
const errors = [];

for (const file of files) {
  const src = fs.readFileSync(file, 'utf8');
  const rel = path.relative(process.cwd(), file).replace(/\\/g, '/');

  // 1. Unused named imports check
  for (const m of src.matchAll(/^import\s+(?:[A-Za-z_$][\w$]*\s*,\s*)?\{([\s\S]*?)\}\s+from\s+['"][^'"]+['"];?/gm)) {
    const body = src.slice(0, m.index) + src.slice(m.index + m[0].length);
    const items = m[1].split(',').map((s) => s.trim()).filter(Boolean);
    for (const item of items) {
      const id = item.split(/\s+as\s+/).pop().trim();
      if (!id) continue;
      const re = new RegExp(`\\b${id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`);
      if (!re.test(body)) {
        errors.push(`${rel}: unused import '${id}'`);
      }
    }
  }

  // 2. JSX undefined component tag check (no-undef)
  if (file.endsWith('.jsx')) {
    const declared = new Set(['React', 'Fragment', 'Suspense', 'StrictMode']);
    for (const m of src.matchAll(/^import\s+([A-Za-z_$][\w$]*)/gm)) declared.add(m[1]);
    for (const m of src.matchAll(/^import\s+(?:[A-Za-z_$][\w$]*\s*,\s*)?\{([\s\S]*?)\}\s+from/gm)) {
      for (const part of m[1].split(',')) {
        const id = part.trim().split(/\s+as\s+/).pop().trim();
        if (id) declared.add(id);
      }
    }
    for (const m of src.matchAll(/\b(?:function|const|let|var|class)\s+([A-Z][\w$]*)\b/g)) {
      declared.add(m[1]);
    }
    // Also allow prop-destructured PascalCase components like `{ icon: Icon }`
    for (const m of src.matchAll(/\b[a-z_][\w$]*\s*:\s*([A-Z][\w$]*)\b/g)) {
      declared.add(m[1]);
    }
    const lines = src.split(/\r?\n/);
    lines.forEach((line, idx) => {
      for (const m of line.matchAll(/<([A-Z][A-Za-z0-9_$]*)(?=[\s/>])/g)) {
        if (!declared.has(m[1])) {
          errors.push(`${rel}:${idx + 1}: undefined JSX component <${m[1]}>`);
        }
      }
    });
  }
}

if (errors.length > 0) {
  console.error(errors.join('\n'));
  process.exit(1);
}
console.log(`[lint] Passed (${files.length} files, 0 unused imports, 0 undefined JSX components).`);
