import shutil
import subprocess
import textwrap

import pytest


def test_strided_tensor_layout_header_indexes_dense_coordinates(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is required")
    source = tmp_path / "tensor_accessor_smoke.cpp"
    binary = tmp_path / "tensor_accessor_smoke"
    source.write_text(
        textwrap.dedent(
            """
            #include <dinoml/tensor_accessor.h>

            int main() {
              const int64_t shape[] = {2, 3, 4};
              const int64_t dense_strides[] = {12, 4, 1};
              const int64_t padded_strides[] = {24, 8, 1};
              const int64_t indices[] = {1, 2, 3};
              dinoml::access::StridedTensorLayout dense{shape, dense_strides, 3, 0};
              dinoml::access::StridedTensorLayout padded{shape, padded_strides, 3, 5};
              if (dense.index(indices) != 23) return 1;
              if (dense.dense_index(23) != 23) return 2;
              if (padded.index(indices) != 48) return 3;
              if (padded.dense_index(23) != 48) return 4;
              return 0;
            }
            """
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", "-Ikernels/common/include", str(source), "-o", str(binary)],
        cwd="/workspace/dinoml_v2",
        check=True,
    )
    subprocess.run([str(binary)], check=True)
