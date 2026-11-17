"""Minimal byte-level GGUF reader/writer for tests (v1/v2/v3 layouts, little-endian).

Keeps every KV value as parsed python data plus its GGUF type, so a file can be re-emitted in another
layout (v1: 32-bit counts / lengths / dims) or with tensors added. Only what the fork's tests need:
no endian swapping, no quant awareness (tensor data is copied as opaque bytes).
"""
import struct

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
_SCALAR = {T_U8: '<B', T_I8: '<b', T_U16: '<H', T_I16: '<h', T_U32: '<I', T_I32: '<i', T_F32: '<f',
           T_BOOL: '<?', T_U64: '<Q', T_I64: '<q', T_F64: '<d'}


class GGUFRaw:
    def __init__(self):
        self.version = 3
        self.kv = []        # (key, type, value) ; arrays: value = (elem_type, [values])
        self.tensors = []   # dict(name, dims (ne list), type, data bytes)
        self.alignment = 32


def _rd(buf, pos, fmt):
    v = struct.unpack_from(fmt, buf, pos)[0]
    return v, pos + struct.calcsize(fmt)


def load(path):
    with open(path, 'rb') as f:
        buf = f.read()
    assert buf[:4] == b'GGUF', 'not a GGUF file'
    g = GGUFRaw()
    pos = 4
    g.version, pos = _rd(buf, pos, '<I')
    cnt = '<I' if g.version == 1 else '<Q'
    n_t, pos = _rd(buf, pos, cnt)
    n_kv, pos = _rd(buf, pos, cnt)

    def rstr(p):
        n, p = _rd(buf, p, cnt)
        return buf[p:p + n].decode('utf-8'), p + n

    def rval(t, p):
        if t == T_STR:
            return rstr(p)
        return _rd(buf, p, _SCALAR[t])

    for _ in range(n_kv):
        key, pos = rstr(pos)
        t, pos = _rd(buf, pos, '<I')
        if t == T_ARR:
            et, pos = _rd(buf, pos, '<I')
            n, pos = _rd(buf, pos, cnt)
            vals = []
            for _ in range(n):
                v, pos = rval(et, pos)
                vals.append(v)
            g.kv.append((key, t, (et, vals)))
        else:
            v, pos = rval(t, pos)
            g.kv.append((key, t, v))
        if key == 'general.alignment':
            g.alignment = g.kv[-1][2]
    infos = []
    for _ in range(n_t):
        name, pos = rstr(pos)
        nd, pos = _rd(buf, pos, '<I')
        dims = []
        for _ in range(nd):
            d, pos = _rd(buf, pos, cnt)
            dims.append(d)
        tt, pos = _rd(buf, pos, '<I')
        off, pos = _rd(buf, pos, '<Q')
        infos.append((name, dims, tt, off))
    data0 = (pos + g.alignment - 1) // g.alignment * g.alignment
    for i, (name, dims, tt, off) in enumerate(infos):
        end = infos[i + 1][3] if i + 1 < len(infos) else len(buf) - data0
        g.tensors.append({'name': name, 'dims': dims, 'type': tt, 'data': buf[data0 + off:data0 + end]})
    return g


def dump(g, path, version=None):
    """Write g. version=1 emits the v1 layout (u32 counts, string lengths, array lengths and dims);
    tensor offsets stay u64 in every version. Each tensor's data block is padded to the alignment."""
    v = g.version if version is None else version
    cnt = '<I' if v == 1 else '<Q'
    out = bytearray(b'GGUF')
    out += struct.pack('<I', v)
    out += struct.pack(cnt, len(g.tensors))
    out += struct.pack(cnt, len(g.kv))

    def wstr(s):
        b = s.encode('utf-8')
        out.extend(struct.pack(cnt, len(b)))
        out.extend(b)

    def wval(t, x):
        if t == T_STR:
            wstr(x)
        else:
            out.extend(struct.pack(_SCALAR[t], x))

    for key, t, val in g.kv:
        wstr(key)
        out += struct.pack('<I', t)
        if t == T_ARR:
            et, vals = val
            out += struct.pack('<I', et)
            out += struct.pack(cnt, len(vals))
            for x in vals:
                wval(et, x)
        else:
            wval(t, val)
    al = g.alignment
    off = 0
    blobs = []
    for t in g.tensors:
        wstr(t['name'])
        out += struct.pack('<I', len(t['dims']))
        for d in t['dims']:
            out += struct.pack(cnt, d)
        out += struct.pack('<I', t['type'])
        out += struct.pack('<Q', off)
        data = bytes(t['data'])
        pad = (-len(data)) % al
        blobs.append(data + b'\0' * pad)
        off += len(data) + pad
    out += b'\0' * ((-len(out)) % al)
    for b in blobs:
        out += b
    with open(path, 'wb') as f:
        f.write(out)
    return len(out)
