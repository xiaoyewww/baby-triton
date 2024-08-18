import torch


class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self._data = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: "torch.Tensor"):
        def _from_dlpack(tensor):
            from tvm.runtime import Device
            from tvm.runtime import ndarray
            try:
                return ndarray.from_dlpack(tensor)
            except RuntimeError:
                pass
            device_type = tensor.device.type
            device_id = tensor.device.index or 0
            return ndarray.array(
                tensor.numpy(),
                device=Device(
                    Device.STR2MASK[device_type],
                    device_id,
                ),
            )
        data = _from_dlpack(data)
        if data.shape != tuple(self.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(self.shape)},"
                             " got {data.shape}")
        if data.dtype != self.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype},"
                             " got {data.dtype}")
        self._data = data

    def __str__(self):
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

# import torch
# a = Tensor(shape=(2, 3), dtype="float32")
# a.data = torch.ones(size=(2, 3), dtype=torch.float32)
# print(a)
# print(a.data)
# print(type(a.data))
