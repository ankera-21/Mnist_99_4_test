import pytest
import torch
from mnist import Net

@pytest.fixture
def model():
    return Net()

def test_total_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params}')
    assert total_params <= 20_000, "Model should have less than 20000 parameters."

def test_batch_normalization(model):
    batch_norm_layers = sum(1 for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d))
    print(f'Batch Normalization Layers: {batch_norm_layers}')
    assert batch_norm_layers > 0, "Model should have Batch Normalization layers."

def test_dropout(model):
    dropout_layers = sum(1 for layer in model.modules() if isinstance(layer, torch.nn.Dropout))
    print(f'DropOut Layers: {dropout_layers}')
    assert dropout_layers > 0, "Model should have DropOut layers."

def test_fully_connected_layer(model):
    fc_layers = sum(1 for layer in model.modules() if isinstance(layer, torch.nn.Linear))
    print(f'Fully Connected Layers: {fc_layers}')
    assert fc_layers > 0, "Model should have Fully Connected layers."

if __name__ == '__main__':
    pytest.main()
