# 延后事件与报告边界的顺序模型和源码门，依据 PR9 及 code9 的完成契约。
# Ordering model and source gates for deferred events/report boundaries, based on PR9 and code9 completion contracts.
from pathlib import Path
from texture_cache_regression import balanced
root=Path(__file__).resolve().parent.parent
s=(root/'xbg_game.cpp').read_text(encoding='utf-8')
h=(root/'../../nvn/hle_nvn.c').resolve().read_text(encoding='utf-8')
report=balanced(s,s.index('case OP_REPORT_COUNTER:'))
assert report.index('Commands c')<report.index('epochs[counter_epoch].size()')
commands=balanced(s,s.index('struct Commands '))
assert 'Commands(bool is_draw=false):draw(is_draw)' in commands
assert 'if(!draw){end_open_pass();full_barrier(command);}' in commands
for name in ['nvnEventInitialize','nvnEventFinalize','nvnEventGetValue']:
    assert 'mtl_event_read' in balanced(h,h.index('NVN('+name+','))
assert 'mtl_event_read' in balanced(h,h.index('static void ev_signal('))
read=balanced(s,s.index('extern "C" void xbg_game_event_read('))
assert read.index('gfx_drain()')<read.index('lock(mutex)')<read.index('complete_all()')
wait=balanced(s,s.index('case OP_COUNT_+4:'))
assert wait.index('complete_all()')<wait.index('__atomic_load_n')
assert 'defer_events=0;if(m)' in s

class Events:
    def __init__(self): self.values={}; self.pending=[]
    def signal_gpu(self,a,m,v): self.pending.append((a,m,v))
    def drain(self):
        for a,m,v in self.pending:
            self.values[a]=(self.values.get(a,0)+v if m=='add' else v)&0xffffffff
        self.pending=[]
    def cpu(self,a,m,v):
        self.drain();self.values[a]=(self.values.get(a,0)+v if m=='add' else v)&0xffffffff
    def read(self,a): self.drain();return self.values.get(a,0)
for first in ['write','add']:
    for second in ['write','add']:
        e=Events();e.values[1]=7;e.signal_gpu(1,first,11);e.cpu(1,second,3)
        expect=11 if first=='write' else 18
        expect=3 if second=='write' else expect+3
        assert e.read(1)==expect
        e.signal_gpu(1,'add',5);e.drain() # Finalize/reinitialize storage boundary.
        e.values[1]=0;e.signal_gpu(1,'write',9);assert e.read(1)==9
        e.signal_gpu(1,'write',0xffffffff);e.cpu(1,'add',2);assert e.read(1)==1
print('PASS: event WRITE/ADD, CPU overwrite, wrap and reused storage model; host hooks and report-end-pass source gates (guest direct reads unproven)')
