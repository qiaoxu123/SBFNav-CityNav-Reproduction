import numpy as np

from sbfnav.coordinates import MapTransform


def test_world_normalized_world_round_trip():
    transform = MapTransform("wide", 100.0, 300.0, 500.0, 500.0)
    points = np.asarray(((100, 300), (500, 500), (250.25, 420.75)))
    reconstructed = transform.normalized_to_world(transform.world_to_normalized(points))
    np.testing.assert_allclose(reconstructed, points, atol=1e-10)


def test_square_padding_and_north_up_semantics():
    transform = MapTransform("wide", 100.0, 300.0, 500.0, 500.0)
    assert transform.padded_bounds == (100.0, 200.0, 500.0, 600.0)
    north_west = transform.world_to_normalized((100.0, 500.0))
    south_east = transform.world_to_normalized((500.0, 300.0))
    np.testing.assert_allclose(north_west, (0.0, 0.25))
    np.testing.assert_allclose(south_east, (1.0, 0.75))


def test_grid_centers_and_quantization_bound():
    transform = MapTransform("square", 0.0, 0.0, 400.0, 400.0)
    centers = transform.field_to_world(np.asarray(((0, 0), (27, 27), (10, 14))))
    continuous = transform.world_to_field(centers)
    np.testing.assert_allclose(continuous, ((0, 0), (27, 27), (10, 14)), atol=1e-12)
    point = np.asarray((173.2, 91.4))
    index = transform.field_index(point, clip=True)
    quantized = transform.field_to_world(index)
    assert np.linalg.norm(point - quantized) <= np.sqrt(2) * transform.field_meters_per_cell / 2


def test_valid_mask_excludes_padding():
    transform = MapTransform("tall", 0.0, 0.0, 200.0, 400.0)
    valid = transform.valid_mask(28)
    assert valid.shape == (28, 28)
    assert not valid[:, :6].any()
    assert valid[:, 8:20].all()
    assert not valid[:, 22:].any()

