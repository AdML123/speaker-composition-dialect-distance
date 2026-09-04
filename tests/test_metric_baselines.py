import torch

from src.metric_baselines import DiagonalMetric


def test_diagonal_metric_has_nonnegative_scales_and_1024_parameters():
    model = DiagonalMetric(1024)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1024
    assert torch.all(model.scales() > 0)


def test_diagonal_metric_preserves_input_dimension():
    model = DiagonalMetric(4)
    inputs = torch.ones((3, 4))
    assert model(inputs).shape == inputs.shape
