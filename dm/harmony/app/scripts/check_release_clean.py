# 检查发布包（.app 或 .hap）里没有内置运行包（resources/resfile/dm/…，从游戏文件生成，只许进调试版）。有就返回 1。
# 用法：python -I scripts/check_release_clean.py <app-release-signed.app>
# Check that a release package (.app or .hap) carries no bundled runtime pack (resources/resfile/dm/..., generated from game files, debug builds only). Exits 1 if it does.
# Usage: python -I scripts/check_release_clean.py <app-release-signed.app>
import io
import sys
import zipfile


def bad_entries(zf):
    found = []
    for name in zf.namelist():
        if name.lower().endswith('.hap'):
            with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
                found += [name + '!' + n for n in bad_entries(inner)]
        elif name.startswith('resources/resfile/dm/'):
            found.append(name)
    return found


with zipfile.ZipFile(sys.argv[1]) as z:
    bad = bad_entries(z)
if bad:
    print('NOT CLEAN: %d bundled pack files, e.g. %s' % (len(bad), bad[0]))
    sys.exit(1)
print('clean: no bundled runtime pack')
