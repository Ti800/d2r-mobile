# 核对生产State完整字段清单及memo生命周期接线；补足ABI padding内新增字段无法靠sizeof发现的情况。
# Check the full production State inventory and memo lifecycle wiring, including new fields hidden inside ABI padding.
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from texture_cache_regression import balanced

root = Path(__file__).resolve().parent.parent
source = (root / 'xbg_game.cpp').read_text(encoding='utf-8')
header = (root / 'xbg_pipeline_memo.h').read_text(encoding='utf-8')
start = header.index('template<class ShaderType> struct XbgPipelineDrawState{')
body = header[header.index('{', start) + 1:header.index('\n};', start)]
while '{' in body:
    next_body = re.sub(r'\{[^{}]*\}', '', body)
    assert next_body != body, 'Cannot parse State initializer; audit the inventory checker'
    body = next_body
body = re.sub(r'\[[^]]*\]|=[^,;]+', '', body)
names = [re.findall(r'[A-Za-z_]\w*', part)[-1]
         for declaration in body.split(';') if declaration.strip()
         for part in declaration.split(',')]
expected = ['rt', 'shader', 'ub', 'sb', 'vb', 'tex', 'samp', 'img', 'texture_pool', 'sampler_pool',
            'attrib', 'stream', 'blend', 'color', 'mask', 'ms', 'poly', 'ds', 'viewport', 'scissor',
            'blend_color', 'bias', 'stencil_compare', 'stencil_write', 'stencil_ref', 'copy_stride',
            'copy_image_stride', 'patch_size', 'bounds', 'count_samples', 'samples_passed']
assert names == expected, ('New or reordered State fields require explicit pipeline/dynamic/resource policy', names)
assert 'using State=XbgPipelineDrawState<Shader>;' in source
for tag in ['struct Shader {XbgPipelineGeneration lifetime;',
            'struct LayoutEntry {XbgPipelineGeneration lifetime;',
            'XbgPipelineGeneration lifetime;std::vector<uint32_t> words;']:
    assert tag in source, tag
definition = balanced(source, source.index('struct PipelineDefinition {'))
assert 'lifetime.refresh()' in definition
for signature, first_destroy in [('void clear_pipeline_objects(){', 'vkDestroyPipeline'),
                                  ('void clear_descriptor_caches(){', 'vkDestroyDescriptorUpdateTemplate'),
                                  ('void clear_render_pass_cache(){', 'vkDestroyRenderPass')]:
    block = balanced(source, source.index(signature))
    assert block.index('pipe_memo_invalidate()') < block.index(first_destroy)
lookup = balanced(source, source.index('void pipeline_lookup('))
assert lookup.index('pipe_memo_invalidate()') < lookup.index('pipeline_objects.erase(oldest)')
assert 'out.entry=&found->second' in lookup
assert 'pipeline.entry->last_used=++pipeline_tick' in source
assert '~Shader(){if(initialized)pipe_memo_invalidate();' in source
assert 'pipe_memo_next;bool pipe_memo=false;' in source
assert '!ab_off("pipememo")' in source and '"/gfx-pipe-memo"' in source
assert 'pipeline_fast' not in source
assert 'std::is_standard_layout<S>::value' in header and 'XBG_PIPE_NEXT' in header
print('PASS: 31 actual State fields, generation identities, clear/evict/destroy invalidation, LRU and opt-in/A-B source wiring; not a native or GPU test')
