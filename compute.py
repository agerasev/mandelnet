import numpy as np
import pyopencl as cl


class _BufferProto:
    def __init__(self, ctx, nparray, rw=cl.mem_flags.READ_WRITE, dual=False, **kwargs):
        self.ctx = ctx
        self.rw = rw
        self.dual = dual
        self.host = nparray
        
        self.buf = self._mkbuf(**kwargs)
        if dual:
            self.dbuf = self._mkbuf(**kwargs)
        
    def swap(self):
        if self.dual:
            self.buf, self.dbuf = self.dbuf, self.buf

    def load(self, queue):
        cl.enqueue_copy(queue, self.host, self.buf, **self._lkws())

class Buffer(_BufferProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _mkbuf(self):
        return cl.Buffer(self.ctx, self.rw | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.host)
    
    def _lkws(self):
        return {}

class _ImageProto(_BufferProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _mkbuf(self, fmt):
        return cl.Image(
            self.ctx, 
            self.rw | cl.mem_flags.COPY_HOST_PTR, 
            fmt, 
            hostbuf=self.host,
            shape=self._shape(),
        )
    
    def _lkws(self):
        sh = self._shape()
        return {
            "origin": np.zeros_like(sh),
            "region": sh,
        }

class Image1D(_ImageProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _shape(self):
        return (self.host.shape[0],)
    
class Image2D(_ImageProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _shape(self):
        return (self.host.shape[1], self.host.shape[0])

class Image3D(_ImageProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _shape(self):
        return (self.host.shape[2], self.host.shape[1], self.host.shape[0])
