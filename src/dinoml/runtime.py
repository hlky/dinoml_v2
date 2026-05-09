from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from dinoml.ir import (
    RUNTIME_ABI_VERSION,
    array_from_storage,
    array_to_storage,
    dtype_numpy,
    dtype_runtime_enum,
    normalize_dtype,
    read_json,
)
from dinoml.constant_sources import constant_source_from_storage
from dinoml.shapes import infer_output_shape, validate_runtime_shape


class _DinoTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("ndim", ctypes.c_size_t),
        ("dtype", ctypes.c_int),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_size_t),
        ("nbytes", ctypes.c_size_t),
        ("device_type", ctypes.c_int),
        ("flags", ctypes.c_uint32),
        ("alignment", ctypes.c_size_t),
    ]


DINO_DEVICE_CPU = 0
DINO_DEVICE_CUDA = 1
DINO_TENSOR_FLAG_CONTIGUOUS = 1


def load(path: str | Path) -> "RuntimeModule":
    return RuntimeModule(Path(path))


class RuntimeModule:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        manifest_path = artifact_dir / "manifest.json"
        module_path = artifact_dir / "module.so"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        if not module_path.exists():
            raise FileNotFoundError(f"Missing module: {module_path}")
        self.manifest = read_json(manifest_path)
        self.target_name = self.manifest.get("target", {}).get("name")
        if self.manifest.get("runtime_abi_version") != RUNTIME_ABI_VERSION:
            raise RuntimeError(
                f"Unsupported runtime ABI {self.manifest.get('runtime_abi_version')}, expected {RUNTIME_ABI_VERSION}"
            )
        files = self.manifest.get("files", {})
        runtime_lib = artifact_dir / files.get("runtime_library", "lib/libdinoml_runtime.so")
        kernels_lib = artifact_dir / files.get("kernel_library", "lib/libdinoml_cuda_kernels.so")
        global_load_mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
        local_load_mode = getattr(os, "RTLD_NOW", 0)
        self._runtime_dll = ctypes.CDLL(str(runtime_lib), mode=global_load_mode)
        self._cuda_runtime_dll = None
        if "cuda_runtime_library" in files:
            cuda_runtime_lib = artifact_dir / files["cuda_runtime_library"]
            self._cuda_runtime_dll = ctypes.CDLL(str(cuda_runtime_lib), mode=global_load_mode)
        self._kernels_dll = ctypes.CDLL(str(kernels_lib), mode=global_load_mode)
        self._dll = ctypes.CDLL(str(module_path), mode=local_load_mode)
        self._configure_symbols()
        if self._runtime_dll.dino_abi_version() != RUNTIME_ABI_VERSION:
            raise RuntimeError("module.so ABI version does not match Python runtime")
        self._handle = ctypes.c_void_p()
        self._check(self._dll.dino_module_load(str(artifact_dir).encode("utf-8"), ctypes.byref(self._handle)))
        metadata_raw = self._dll.dino_module_get_metadata_json(self._handle)
        self.metadata = json.loads(metadata_raw.decode("utf-8"))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._check(self._dll.dino_module_free(self._handle))
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def create_session(self) -> "Session":
        return Session(self)

    def load_constants_from_file(self, path: str | Path | None = None) -> None:
        constant_path = self.artifact_dir / "constants.bin" if path is None else Path(path)
        self._check(self._dll.dino_module_load_constants(self._handle, str(constant_path).encode("utf-8")))

    def load_encoded_constants(self) -> None:
        for constant_spec in self.metadata["constants"]:
            storage = constant_spec.get("storage")
            if not isinstance(storage, Mapping):
                continue
            source = constant_source_from_storage(storage)
            if source is None:
                continue
            materialized = source.materialize(str(constant_spec["dtype"]), constant_spec["shape"])
            self.set_constant_numpy(str(constant_spec["name"]), materialized.array)

    def set_constant_numpy(self, name: str, value: np.ndarray) -> None:
        constants = {constant["name"]: constant for constant in self.metadata["constants"]}
        if name not in constants:
            raise ValueError(f"Unknown constant: {name}")
        constant_spec = constants[name]
        array = array_to_storage(value, str(constant_spec["dtype"]))
        actual_shape = validate_runtime_shape(name, array.shape, constant_spec)
        dtype_enum = dtype_runtime_enum(constant_spec["dtype"])
        if self.target_name == "cpu":
            tensor, keepalive = _make_dino_tensor(
                ctypes.c_void_p(array.ctypes.data),
                actual_shape,
                dtype_enum,
                nbytes=array.nbytes,
                device_type=DINO_DEVICE_CPU,
            )
            self._check(
                self._dll.dino_module_set_constant(
                    self._handle,
                    name.encode("utf-8"),
                    ctypes.byref(tensor),
                )
            )
            return
        if self._cuda_runtime_dll is None:
            raise RuntimeError("CUDA runtime library is not loaded")
        ptr = ctypes.c_void_p()
        self._check(self._cuda_runtime_dll.dino_device_malloc(ctypes.byref(ptr), ctypes.c_size_t(array.nbytes)))
        tensor, keepalive = _make_dino_tensor(
            ptr,
            actual_shape,
            dtype_enum,
            nbytes=array.nbytes,
            device_type=DINO_DEVICE_CUDA,
        )
        try:
            self._check(
                self._cuda_runtime_dll.dino_copy_host_to_device(
                    ptr,
                    ctypes.c_void_p(array.ctypes.data),
                    ctypes.c_size_t(array.nbytes),
                )
            )
            self._check(
                self._dll.dino_module_set_constant(
                    self._handle,
                    name.encode("utf-8"),
                    ctypes.byref(tensor),
                )
            )
        finally:
            self._check(self._cuda_runtime_dll.dino_device_free(ptr))

    def set_constant_device_pointer(self, name: str, ptr: int, shape: tuple[int, ...] | list[int], dtype: str) -> None:
        if self.target_name != "cuda":
            raise RuntimeError("set_constant_device_pointer is only available for CUDA artifacts")
        constants = {constant["name"]: constant for constant in self.metadata["constants"]}
        if name not in constants:
            raise ValueError(f"Unknown constant: {name}")
        constant_spec = constants[name]
        actual_shape = validate_runtime_shape(name, shape, constant_spec)
        normalized_dtype = normalize_dtype(str(dtype))
        expected_dtype = str(constant_spec["dtype"])
        if normalized_dtype != expected_dtype:
            raise ValueError(f"Constant {name} has dtype {normalized_dtype}, expected {expected_dtype}")
        nbytes = _shape_nbytes(actual_shape, normalized_dtype)
        tensor, keepalive = _make_dino_tensor(
            ctypes.c_void_p(int(ptr)),
            actual_shape,
            dtype_runtime_enum(normalized_dtype),
            nbytes=nbytes,
            device_type=DINO_DEVICE_CUDA,
        )
        self._check(self._dll.dino_module_set_constant(self._handle, name.encode("utf-8"), ctypes.byref(tensor)))

    def set_constant_torch(self, name: str, value: object) -> None:
        if not getattr(value, "is_cuda", False):
            raise ValueError(f"Constant {name} must be a CUDA tensor")
        if not value.is_contiguous():
            raise ValueError(f"Constant {name} must be contiguous")
        spec = {constant["name"]: constant for constant in self.metadata["constants"]}[name]
        actual_shape = validate_runtime_shape(name, tuple(int(dim) for dim in value.shape), spec)
        self.set_constant_device_pointer(
            name,
            _torch_data_ptr(value),
            actual_shape,
            _torch_dtype_name(value),
        )

    def _configure_symbols(self) -> None:
        self._runtime_dll.dino_abi_version.restype = ctypes.c_int
        self._runtime_dll.dino_get_last_error.restype = ctypes.c_char_p
        if self._cuda_runtime_dll is not None:
            self._cuda_runtime_dll.dino_device_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self._cuda_runtime_dll.dino_device_malloc.restype = ctypes.c_int
            self._cuda_runtime_dll.dino_device_free.argtypes = [ctypes.c_void_p]
            self._cuda_runtime_dll.dino_device_free.restype = ctypes.c_int
            self._cuda_runtime_dll.dino_copy_host_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            self._cuda_runtime_dll.dino_copy_host_to_device.restype = ctypes.c_int
            self._cuda_runtime_dll.dino_copy_device_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            self._cuda_runtime_dll.dino_copy_device_to_host.restype = ctypes.c_int
            self._cuda_runtime_dll.dino_copy_device_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            self._cuda_runtime_dll.dino_copy_device_to_device.restype = ctypes.c_int
        self._dll.dino_module_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.dino_module_load.restype = ctypes.c_int
        self._dll.dino_module_free.argtypes = [ctypes.c_void_p]
        self._dll.dino_module_free.restype = ctypes.c_int
        self._dll.dino_module_get_metadata_json.argtypes = [ctypes.c_void_p]
        self._dll.dino_module_get_metadata_json.restype = ctypes.c_char_p
        self._dll.dino_module_load_constants.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._dll.dino_module_load_constants.restype = ctypes.c_int
        self._dll.dino_module_set_constant.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_DinoTensor)]
        self._dll.dino_module_set_constant.restype = ctypes.c_int
        self._dll.dino_session_create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.dino_session_create.restype = ctypes.c_int
        self._dll.dino_session_destroy.argtypes = [ctypes.c_void_p]
        self._dll.dino_session_destroy.restype = ctypes.c_int
        self._dll.dino_session_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._dll.dino_session_set_stream.restype = ctypes.c_int
        self._dll.dino_session_get_output_shape.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._dll.dino_session_get_output_shape.restype = ctypes.c_int
        self._dll.dino_session_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_DinoTensor),
            ctypes.c_size_t,
            ctypes.POINTER(_DinoTensor),
            ctypes.c_size_t,
        ]
        self._dll.dino_session_run.restype = ctypes.c_int

    def _check(self, err: int) -> None:
        if err:
            message = self._last_error_message()
            raise RuntimeError(message.decode("utf-8") if message else "Unknown DinoML runtime error")

    def _last_error_message(self) -> bytes | None:
        getters = []
        for dll in (getattr(self, "_dll", None), getattr(self, "_runtime_dll", None)):
            if dll is None:
                continue
            try:
                getter = dll.dino_get_last_error
            except AttributeError:
                continue
            getter.restype = ctypes.c_char_p
            getters.append(getter)
        try:
            global_getter = ctypes.CDLL(None).dino_get_last_error
            global_getter.restype = ctypes.c_char_p
            getters.append(global_getter)
        except AttributeError:
            pass
        for getter in getters:
            message = getter()
            if message:
                return message
        return None


class Session:
    def __init__(self, module: RuntimeModule):
        self.module = module
        self._handle = ctypes.c_void_p()
        self._cuda_buffers: Dict[str, tuple[ctypes.c_void_p, int]] = {}
        self.module._check(self.module._dll.dino_session_create(self.module._handle, ctypes.byref(self._handle)))

    def close(self) -> None:
        self._free_cuda_buffers()
        if getattr(self, "_handle", None):
            self.module._check(self.module._dll.dino_session_destroy(self._handle))
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def run_numpy(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.module.target_name == "cpu":
            return self._run_numpy_cpu(inputs)
        return self._run_numpy_cuda(inputs)

    def set_stream(self, stream: object | None) -> None:
        self.module._check(self.module._dll.dino_session_set_stream(self._handle, _as_c_void_p(stream)))

    def get_output_shape(self, index_or_name: int | str) -> tuple[int, ...]:
        output_index = self._output_index(index_or_name)
        ndim = ctypes.c_size_t(0)
        self.module._check(
            self.module._dll.dino_session_get_output_shape(
                self._handle,
                ctypes.c_size_t(output_index),
                None,
                ctypes.byref(ndim),
            )
        )
        shape = (ctypes.c_int64 * ndim.value)()
        self.module._check(
            self.module._dll.dino_session_get_output_shape(
                self._handle,
                ctypes.c_size_t(output_index),
                shape,
                ctypes.byref(ndim),
            )
        )
        return tuple(int(shape[i]) for i in range(ndim.value))

    def run_device_pointers(
        self,
        inputs: Mapping[str, int],
        outputs: Mapping[str, int],
        input_shapes: Mapping[str, tuple[int, ...] | list[int]] | None = None,
        output_shapes: Mapping[str, tuple[int, ...] | list[int]] | None = None,
    ) -> None:
        if self.module.target_name != "cuda":
            raise RuntimeError("run_device_pointers is only available for CUDA artifacts")
        input_specs = self.module.metadata["inputs"]
        output_specs = self.module.metadata["outputs"]
        shape_buffers = []
        resolved_input_shapes: dict[str, tuple[int, ...]] = {}
        input_tensors = (_DinoTensor * len(input_specs))()
        for idx, spec in enumerate(input_specs):
            name = str(spec["name"])
            if name not in inputs:
                raise ValueError(f"Missing input pointer: {name}")
            actual_shape = tuple(int(dim) for dim in (input_shapes or {}).get(name, spec["shape"]))
            validate_runtime_shape(name, actual_shape, spec)
            resolved_input_shapes[name] = actual_shape
            tensor, keepalive = _make_dino_tensor(
                ctypes.c_void_p(int(inputs[name])),
                actual_shape,
                dtype_runtime_enum(str(spec["dtype"])),
                nbytes=_shape_nbytes(actual_shape, str(spec["dtype"])),
                device_type=DINO_DEVICE_CUDA,
            )
            shape_buffers.extend(keepalive)
            input_tensors[idx] = tensor
        output_tensors = (_DinoTensor * len(output_specs))()
        for idx, spec in enumerate(output_specs):
            name = str(spec["name"])
            if name not in outputs:
                raise ValueError(f"Missing output pointer: {name}")
            if output_shapes is not None and name in output_shapes:
                actual_shape = tuple(int(dim) for dim in output_shapes[name])
            else:
                actual_shape = infer_output_shape(spec, input_specs, resolved_input_shapes)
            validate_runtime_shape(name, actual_shape, spec)
            tensor, keepalive = _make_dino_tensor(
                ctypes.c_void_p(int(outputs[name])),
                actual_shape,
                dtype_runtime_enum(str(spec["dtype"])),
                nbytes=_shape_nbytes(actual_shape, str(spec["dtype"])),
                device_type=DINO_DEVICE_CUDA,
            )
            shape_buffers.extend(keepalive)
            output_tensors[idx] = tensor
        self.module._check(
            self.module._dll.dino_session_run(
                self._handle,
                input_tensors,
                ctypes.c_size_t(len(input_specs)),
                output_tensors,
                ctypes.c_size_t(len(output_specs)),
            )
        )

    def run_torch(self, inputs: Mapping[str, object]) -> Dict[str, object]:
        import torch

        if self.module.target_name != "cuda":
            raise RuntimeError("run_torch is only available for CUDA artifacts")
        input_specs = self.module.metadata["inputs"]
        output_specs = self.module.metadata["outputs"]
        input_shapes = {}
        for spec in input_specs:
            tensor = inputs[str(spec["name"])]
            if not getattr(tensor, "is_cuda", False):
                raise ValueError(f"Input {spec['name']} must be a CUDA tensor")
            validate_runtime_shape(str(spec["name"]), tuple(int(dim) for dim in tensor.shape), spec)
            input_shapes[str(spec["name"])] = tuple(int(dim) for dim in tensor.shape)
            if _torch_dtype_name(tensor) != str(spec["dtype"]):
                raise ValueError(f"Input {spec['name']} has dtype {_torch_dtype_name(tensor)}, expected {spec['dtype']}")
            if not tensor.is_contiguous():
                raise ValueError(f"Input {spec['name']} must be contiguous")
        device = next(iter(inputs.values())).device
        outputs = {
            str(spec["name"]): torch.empty(
                infer_output_shape(spec, input_specs, input_shapes),
                dtype=_torch_dtype(str(spec["dtype"]), torch),
                device=device,
            )
            for spec in output_specs
        }
        self.run_device_pointers(
            {str(spec["name"]): _torch_data_ptr(inputs[str(spec["name"])]) for spec in input_specs},
            {name: _torch_data_ptr(tensor) for name, tensor in outputs.items()},
            {str(spec["name"]): tuple(int(dim) for dim in inputs[str(spec["name"])].shape) for spec in input_specs},
            {name: tuple(int(dim) for dim in tensor.shape) for name, tensor in outputs.items()},
        )
        return outputs

    def _run_numpy_cpu(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        input_specs = self.module.metadata["inputs"]
        output_specs = self.module.metadata["outputs"]
        input_arrays = [_prepare_input(spec, inputs) for spec in input_specs]
        input_shapes = {str(spec["name"]): array.shape for spec, array in zip(input_specs, input_arrays)}
        output_arrays = [
            np.empty(infer_output_shape(spec, input_specs, input_shapes), dtype=dtype_numpy(str(spec["dtype"])))
            for spec in output_specs
        ]
        shape_buffers = []
        input_tensors = (_DinoTensor * len(input_arrays))()
        for idx, (spec, array) in enumerate(zip(input_specs, input_arrays)):
            tensor, keepalive = _make_dino_tensor(
                ctypes.c_void_p(array.ctypes.data),
                array.shape,
                dtype_runtime_enum(str(spec["dtype"])),
                nbytes=array.nbytes,
                device_type=DINO_DEVICE_CPU,
            )
            shape_buffers.extend(keepalive)
            input_tensors[idx] = tensor
        output_tensors = (_DinoTensor * len(output_arrays))()
        for idx, (spec, array) in enumerate(zip(output_specs, output_arrays)):
            tensor, keepalive = _make_dino_tensor(
                ctypes.c_void_p(array.ctypes.data),
                array.shape,
                dtype_runtime_enum(str(spec["dtype"])),
                nbytes=array.nbytes,
                device_type=DINO_DEVICE_CPU,
            )
            shape_buffers.extend(keepalive)
            output_tensors[idx] = tensor
        self.module._check(
            self.module._dll.dino_session_run(
                self._handle,
                input_tensors,
                ctypes.c_size_t(len(input_arrays)),
                output_tensors,
                ctypes.c_size_t(len(output_arrays)),
            )
        )
        return {spec["name"]: array_from_storage(array, str(spec["dtype"])) for spec, array in zip(output_specs, output_arrays)}

    def _run_numpy_cuda(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        input_specs = self.module.metadata["inputs"]
        output_specs = self.module.metadata["outputs"]
        input_arrays = [_prepare_input(spec, inputs) for spec in input_specs]
        input_shapes = {str(spec["name"]): array.shape for spec, array in zip(input_specs, input_arrays)}
        output_arrays = [
            np.empty(infer_output_shape(spec, input_specs, input_shapes), dtype=dtype_numpy(str(spec["dtype"])))
            for spec in output_specs
        ]

        shape_buffers = []
        input_tensors = (_DinoTensor * len(input_arrays))()
        for idx, (spec, array) in enumerate(zip(input_specs, input_arrays)):
            ptr = self._device_buffer(f"input:{spec['name']}", array.nbytes)
            self._copy_h2d(ptr, array)
            tensor, keepalive = _make_dino_tensor(
                ptr,
                array.shape,
                dtype_runtime_enum(str(spec["dtype"])),
                nbytes=array.nbytes,
                device_type=DINO_DEVICE_CUDA,
            )
            shape_buffers.extend(keepalive)
            input_tensors[idx] = tensor

        output_tensors = (_DinoTensor * len(output_arrays))()
        for idx, (spec, array) in enumerate(zip(output_specs, output_arrays)):
            ptr = self._device_buffer(f"output:{spec['name']}", array.nbytes)
            tensor, keepalive = _make_dino_tensor(
                ptr,
                array.shape,
                dtype_runtime_enum(str(spec["dtype"])),
                nbytes=array.nbytes,
                device_type=DINO_DEVICE_CUDA,
            )
            shape_buffers.extend(keepalive)
            output_tensors[idx] = tensor

        self.module._check(
            self.module._dll.dino_session_run(
                self._handle,
                input_tensors,
                ctypes.c_size_t(len(input_arrays)),
                output_tensors,
                ctypes.c_size_t(len(output_arrays)),
            )
        )
        for tensor, array in zip(output_tensors, output_arrays):
            self._copy_d2h(array, tensor.data)
        return {spec["name"]: array_from_storage(array, str(spec["dtype"])) for spec, array in zip(output_specs, output_arrays)}

    def _device_malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.module._check(self.module._cuda_runtime_dll.dino_device_malloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return ptr

    def _device_buffer(self, key: str, nbytes: int) -> ctypes.c_void_p:
        cached = self._cuda_buffers.get(key)
        if cached is not None and cached[1] >= nbytes:
            return cached[0]
        if cached is not None:
            self.module._check(self.module._cuda_runtime_dll.dino_device_free(cached[0]))
        ptr = self._device_malloc(nbytes)
        self._cuda_buffers[key] = (ptr, nbytes)
        return ptr

    def _free_cuda_buffers(self) -> None:
        if not getattr(self, "_cuda_buffers", None):
            return
        if self.module._cuda_runtime_dll is None:
            self._cuda_buffers.clear()
            return
        for ptr, _nbytes in reversed(list(self._cuda_buffers.values())):
            self.module._check(self.module._cuda_runtime_dll.dino_device_free(ptr))
        self._cuda_buffers.clear()

    def _copy_h2d(self, dst_device: ctypes.c_void_p, src: np.ndarray) -> None:
        self.module._check(
            self.module._cuda_runtime_dll.dino_copy_host_to_device(dst_device, ctypes.c_void_p(src.ctypes.data), ctypes.c_size_t(src.nbytes))
        )

    def _copy_d2h(self, dst: np.ndarray, src_device: ctypes.c_void_p) -> None:
        self.module._check(
            self.module._cuda_runtime_dll.dino_copy_device_to_host(ctypes.c_void_p(dst.ctypes.data), src_device, ctypes.c_size_t(dst.nbytes))
        )

    def _output_index(self, index_or_name: int | str) -> int:
        output_specs = self.module.metadata["outputs"]
        if isinstance(index_or_name, str):
            for idx, spec in enumerate(output_specs):
                if str(spec["name"]) == index_or_name:
                    return idx
            raise ValueError(f"Unknown output: {index_or_name}")
        index = int(index_or_name)
        if index < 0 or index >= len(output_specs):
            raise IndexError(f"Output index out of range: {index}")
        return index


def _prepare_input(spec: Mapping[str, object], inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    name = str(spec["name"])
    if name not in inputs:
        raise ValueError(f"Missing input: {name}")
    array = array_to_storage(inputs[name], str(spec["dtype"]))
    validate_runtime_shape(name, array.shape, spec)
    return array


def _shape_buffer(shape: object) -> ctypes.Array:
    values = [int(dim) for dim in shape]
    return (ctypes.c_int64 * len(values))(*values)


def _contiguous_strides(shape: object) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape)
    strides = []
    stride = 1
    for dim in reversed(dims):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _stride_buffer(shape: object) -> ctypes.Array:
    values = _contiguous_strides(shape)
    return (ctypes.c_int64 * len(values))(*values)


def _shape_nbytes(shape: object, dtype: str) -> int:
    nbytes = dtype_numpy(dtype).itemsize
    for dim in shape:
        nbytes *= int(dim)
    return int(nbytes)


def _make_dino_tensor(
    data: ctypes.c_void_p,
    shape: object,
    dtype_enum: int,
    *,
    nbytes: int,
    device_type: int,
) -> tuple[_DinoTensor, tuple[ctypes.Array, ctypes.Array]]:
    actual_shape = tuple(int(dim) for dim in shape)
    shape_buffer = _shape_buffer(actual_shape)
    stride_buffer = _stride_buffer(actual_shape)
    tensor = _DinoTensor(
        data,
        shape_buffer,
        len(actual_shape),
        int(dtype_enum),
        stride_buffer,
        0,
        int(nbytes),
        int(device_type),
        DINO_TENSOR_FLAG_CONTIGUOUS,
        _pointer_alignment(data),
    )
    return tensor, (shape_buffer, stride_buffer)


def _pointer_alignment(ptr: ctypes.c_void_p) -> int:
    value = int(ptr.value or 0)
    if value == 0:
        return 0
    return min(value & -value, 256)


def _as_c_void_p(value: object | None) -> ctypes.c_void_p:
    if value is None:
        return ctypes.c_void_p()
    if isinstance(value, ctypes.c_void_p):
        return value
    if hasattr(value, "cuda_stream"):
        return ctypes.c_void_p(int(value.cuda_stream))
    if hasattr(value, "value"):
        return ctypes.c_void_p(int(value.value or 0))
    try:
        return ctypes.c_void_p(int(value))
    except (TypeError, ValueError):
        return ctypes.cast(value, ctypes.c_void_p)


def _torch_data_ptr(tensor: object) -> int:
    if not hasattr(tensor, "data_ptr"):
        raise TypeError("Expected a torch.Tensor-like object with data_ptr()")
    return int(tensor.data_ptr())


def _torch_dtype_name(tensor: object) -> str:
    dtype_name = str(getattr(tensor, "dtype", ""))
    if dtype_name == "torch.float16":
        return "float16"
    if dtype_name == "torch.float32":
        return "float32"
    if dtype_name == "torch.bfloat16":
        return "bfloat16"
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _torch_dtype(dtype: str, torch_module: object) -> object:
    if dtype == "float16":
        return torch_module.float16
    if dtype == "float32":
        return torch_module.float32
    if dtype == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype}")
