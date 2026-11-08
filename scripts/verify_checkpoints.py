"""Check official checkpoints against pinned SHA-256 digests before anything loads them.

Checkpoints are pickles, and ua_pose.compat allows full unpickling because MMEngine's
format needs it, so integrity is checked first.

The SimCC-R50 digest does not start with the hash in its OpenMMLab file name (8e0f5b59).
Checked on 2026-11-05: its MD5 equals the server ETag and Content-MD5 header, a second
download is byte-identical, the server copy is dated after the name (2022-10-10 vs
2022-09-19), and its pickle imports only the same modules as the checkpoints that do
match. So the file is what OpenMMLab serves and is pinned by its actual digest.
"""
import hashlib
import sys
from pathlib import Path

PINNED_PREFIX = {
    'simcc_res50_256x192.pth':
        'a6be8208dafc3f5a6edc0789fdad2b5b576f48dbaf79131fee7b4a6e261bce43',
    'simcc_vipnas_mbv3_256x192.pth': '719f3489',
    'td_hm_res50_256x192.pth': '04af38ce',
}


def main(ckpt_dir: str = 'checkpoints') -> int:
    failed = 0
    for name, prefix in PINNED_PREFIX.items():
        path = Path(ckpt_dir) / name
        if not path.exists():
            print(f'MISSING  {name}')
            failed += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = digest.startswith(prefix)
        print(f'{"OK" if ok else "BAD":8} {name} sha256={digest[:16]}')
        failed += 0 if ok else 1
    return failed


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
