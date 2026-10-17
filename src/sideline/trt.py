"""Minimal TensorRT engine builder and runner (TensorRT 10 API, torch for buffers)."""
from pathlib import Path

import torch


def build_engine(onnx_path, engine_path, fp16=True, min_batch=1, opt_batch=16, max_batch=64):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    for i in range(net.num_inputs):
        t = net.get_input(i)
        shp = list(t.shape)[1:]
        prof.set_shape(t.name, [min_batch, *shp], [opt_batch, *shp], [max_batch, *shp])
    cfg.add_optimization_profile(prof)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    Path(engine_path).write_bytes(bytes(blob))
    return engine_path


class TrtModel:
    """One input, one output, dynamic batch. __call__(torch cuda tensor) -> torch cuda tensor."""

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt
        self.trt = trt
        self.device = torch.device(device)
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = trt.TensorIOMode.INPUT
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == mode)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) != mode)
        self.max_batch = self.engine.get_tensor_profile_shape(self.inp, 0)[2][0]
        self.stream = torch.cuda.Stream(self.device)

    def _run(self, x):
        x = x.contiguous().float()
        self.ctx.set_input_shape(self.inp, tuple(x.shape))
        y = torch.empty(tuple(self.ctx.get_tensor_shape(self.out)), dtype=torch.float32, device=self.device)
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.out, y.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        return y

    def __call__(self, x):
        if len(x) <= self.max_batch:
            return self._run(x)
        return torch.cat([self._run(x[i:i + self.max_batch]) for i in range(0, len(x), self.max_batch)])
