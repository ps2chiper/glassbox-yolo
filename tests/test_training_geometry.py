from types import SimpleNamespace

import numpy as np
import torch

from yolo.core.modules.detect import Detect
from yolo.data.dataset import YoloDataset
from yolo.engine.utils import TaskAlignedAssigner


def test_uniform_distance_logits_decode_in_each_levels_pixel_grid():
    head = Detect(2, [8, 16])
    features = [torch.zeros(1, head.no, 1, 2), torch.zeros(1, head.no, 1, 1)]
    decoded = head.process_for_inference(features)[0]
    # Uniform logits on bins 0..15 give 7.5 cells of distance on every side.
    expected = torch.tensor([[4, 12, 8], [4, 4, 8], [120, 120, 240], [120, 120, 240]])
    torch.testing.assert_close(decoded[:4], expected.float())
    torch.testing.assert_close(decoded[4:], torch.full((2, 3), 0.5))


def test_small_box_expansion_preserves_strict_edges_and_validity_mask():
    boxes = torch.tensor([[[9., 0., 11., 8.], [9., 0., 11., 8.]]])
    points = torch.tensor([[2., 4.], [3., 4.], [10., 8.], [10., 4.]])
    actual = TaskAlignedAssigner().get_candidates_in_targets(
        boxes, points, torch.tensor([[[True], [False]]]), expand_small_boxes=True
    )
    torch.testing.assert_close(actual, torch.tensor([[[0., 1., 0., 1.], [0., 0., 0., 1.]]]))


def test_rectangular_batches_include_partial_last_batch():
    shapes = [(100, 25), (40, 100), (100, 100), (20, 100), (200, 100)]
    dataset = SimpleNamespace(dataset={"labels": [{"img_shape": s} for s in shapes]},
                              length_of_labels=len(shapes), imgsz=320)
    YoloDataset.set_rect_shapes(dataset, batch_size=2)
    assert dataset.indices == [3, 1, 2, 4, 0]
    np.testing.assert_array_equal(dataset.batch_shapes, [[128, 320], [320, 320], [320, 96]])
