import torch 
import torch_mlir 
from torch_mlir import fx
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)
    
def main():
    # Create an instance of the model
    model = SimpleModel()
    
    # Set the model to evaluation
    example_input = torch.randn(1, 10)
    
    mlir_module =fx.export_and_import(
        model,
        example_input,
        output_type="raw"
    )
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(mlir_module)
    print(compiled)
    
main()