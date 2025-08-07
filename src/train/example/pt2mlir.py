import torch
import torch_mlir
from torch_mlir import ir
from torch_mlir.compiler_utils import run_pipeline_with_repro_report, lower_mlir_module, OutputType
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # 池化到 1x1
        self.fc1 = nn.Linear(64, 128)  # 64 是通道数
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.adaptive_pool(self.relu(self.conv1(x)))
        x = self.adaptive_pool(self.relu(self.conv2(x)))
        # x = self.adaptive_pool(x)  # 池化到 [batch_size, 64, 1, 1]
        x = x.view(x.size(0), -1)  # 展平为 [batch_size, 64]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # Create an instance of the model
    # 我们将使用 ResNet-32，并设置评估模式以禁用 Dropout 和 BatchNorm 更新
    model = CNN()
    model.eval()
    # ResNet-32 期望的输入尺寸是 (batch_size, channels, height, width)
    # 对于 CIFAR-10，即 (1, 3, 32, 32)
    example_input = torch.randn(1, 3, 32, 32)

    # 1. Export the PyTorch model to an ExportedProgram
    prog = torch.export.export(model, (example_input,))

    # 2. Run decompositions to prepare for import
    decomposition_table = get_decomposition_table()
    if decomposition_table:
        prog = prog.run_decompositions(decomposition_table)

    # 3. Import the program into an MLIR module
    context = ir.Context()
    torch_mlir.dialects.torch.register_dialect(context)
    fx_importer = FxImporter(context=context)
    fx_importer.import_frozen_program(prog)
    mlir_module = fx_importer.module
    
    # print("============= MLIR before BitflipHardening =============")
    # print(mlir_module)

    # 4. Define and run the custom pass pipeline including BitflipHardening
    # The pass name is 'torch-bitflip-hardening' as found in Passes.td
    pipeline = (
        "builtin.module("
        "torch-bitflip-hardening,"
        "func.func(torch-match-quantized-custom-ops),"
        "torchdynamo-export-to-torch-backend-pipeline"
        ")"
    )
    run_pipeline_with_repro_report(
        mlir_module,
        pipeline,
        "Lowering TorchFX IR -> Torch Backend IR with BitflipHardening",
    )

    # 5. Lower the module to the desired output type (e.g., linalg-on-tensors)
    final_module = lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, mlir_module)

    # 6. Compile with the backend
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(final_module)
    fx_module = backend.load(compiled)
    output = torch.from_numpy(getattr(fx_module,'main')(example_input.cpu().numpy()))
    print(output)

if __name__ == "__main__":
    main()
