from pytest import approx, fixture, raises

import numpy as np
import pandas as pd

from solartoolbox import spatial


@fixture
def olat():
    # Oldenburg latitude
    olat = 53.143452
    return olat


@fixture
def olon():
    # Oldenburg longitude
    olon = 8.214571
    return olon


@fixture
def dlat():
    # These shifts are only approximately 1 40m cell distance,
    # so we assert relative to a buffer
    dlat = 3.64e-4
    return dlat


@fixture
def dlon():
    # These shifts are only approximately 1 40m cell distance,
    # so we assert relative to a buffer
    dlon = 6e-4
    return dlon


@fixture
def xypts():
    ys = [0, 0, 0, 1, -1, 1, -4]
    xs = [9, 8, 10, 9, 9, 10, 12]
    return xs, ys


@fixture
def projected_dxdy():
    expect = [[0, 0],
              [-1, 0],
              [1, 0],
              [0, 1],
              [0, -1],
              [1, 1],
              [3, -4]]
    return expect


@fixture
def projected_dists():
    c45 = np.cos(np.deg2rad(45))
    expect = {(1, 0): [0, -1, 1, 0, 0, 1, 3],
              (0, 1): [0, 0, 0, 1, -1, 1, -4],
              (1, 1): [0, -c45, c45, c45, -c45, 2*c45,
                       3*c45 - 4*c45]}
    return expect


def test__calc_zone():
    # Outer limits
    assert spatial._calc_zone(-179) == approx(1)
    assert spatial._calc_zone(-180) == approx(1)
    assert spatial._calc_zone(179) == approx(60)
    assert spatial._calc_zone(180) == approx(60)

    # Boundaries of Zone 32
    assert spatial._calc_zone(6) == approx(32)
    assert spatial._calc_zone(9) == approx(32)
    assert spatial._calc_zone(12) == approx(33)


def test_latlon2utm_sitecompare(olat, olon):
    # Value compared with MATLAB known good numbers
    (E, N, zone) = spatial.latlon2utm(olat, olon)
    assert (E, N, zone) == approx((447464.9005, 5888516.6988, 32))


def test_latlon2utm_failure(olat, olon):
    latarray = np.array([olat, olat, olat])
    lonarray = np.array([olon, olon, olon])
    with raises(ValueError):
        (E, N, zone) = spatial.latlon2utm(latarray, lonarray[0:2])
    with raises(ValueError):
        (E, N, zone) = spatial.latlon2utm(latarray[0:2], lonarray)


def test_latlon2utm_arrayin(olat, olon):
    # Value compared with MATLAB known good numbers
    latarray = np.array([olat, olat, olat])
    lonarray = np.array([olon, olon, olon])
    enlist = spatial.latlon2utm(latarray, lonarray)
    for EN in enlist:
        assert(EN[0], EN[1]) == approx((447464.9005, 5888516.6988))


def test_latlon2utm_pandas(olat, olon):
    latarray = np.array([olat, olat, olat])
    lonarray = np.array([olon, olon, olon])
    ind = [1, 2, 3]
    latseries = pd.Series(latarray, index=ind)
    lonseries = pd.Series(lonarray, index=ind)
    enframe = spatial.latlon2utm(latseries, lonseries)
    assert isinstance(enframe, pd.DataFrame)
    assert (enframe.columns == ['E', 'N', 'zone']).all()
    assert (enframe.index == ind).all()
    for enval in np.array(enframe):
        assert (enval[0], enval[1], enval[2]) \
               == approx((447464.9005, 5888516.6988, 32))


def test_latlon2utm_theory():
    # Test theoretically known point for zone 32
    (E, N, zone) = spatial.latlon2utm(0, 9)
    assert (E, N, zone) == approx((500000, 0, 32))

    # Test theoretically known point for zone 1
    (E, N, zone) = spatial.latlon2utm(0, -177)
    assert (E, N, zone) == approx((500000, 0, 1))

    # Test theoretically known point for zone 32
    (E, N, zone) = spatial.latlon2utm(0, 9, zone=32)
    assert (E, N, zone) == approx((500000, 0, 32))

    # Test theoretically known point for zone 1
    (E, N, zone) = spatial.latlon2utm(0, -177, zone=1)
    assert (E, N, zone) == approx((500000, 0, 1))


def test_latlon2utm_32north_defn():
    # Coords from https://epsg.io/32632

    # Bottom Left Corner (westmost extent)
    (E, N, zone) = spatial.latlon2utm(0, 6)
    assert (E, N, zone) == approx((166021.44, 0, 32))

    # Center
    (E, N, zone) = spatial.latlon2utm(84 / 2, 9)
    assert (E, N, zone) == approx((500000, 4649776.22, 32))

    # Top Right Corner (also northmost extent)
    (E, N, zone) = spatial.latlon2utm(84, 12 - 1e-10)
    assert (E, N, zone) == approx((534994.66, 9329005.18, 32))

    # Top Left Corner (should also be northmost extent)
    (E, N, _) = spatial.latlon2utm(84, 6)
    assert N == approx(9329005.18)


def test_latlon2utm_32south_defn():
    # Test for the southern hemisphere
    # Coords from https://epsg.io/32732

    # Bottom Left Corner
    (E, N, zone) = spatial.latlon2utm(-80, 6)
    assert (E, N, zone) == approx((441867.78, 1116915.04, 32))

    # Center
    (E, N, zone) = spatial.latlon2utm(-80 / 2, 9)
    assert (E, N, zone) == approx((500000, 5572242.78, 32))

    # Top Right Corner
    (E, N, zone) = spatial.latlon2utm(0 - 1e-12, 12 - 1e-10)
    assert (E, N, zone) == approx((833978.56, 10000000.00, 32))

    # Bottom Right Corner (should also be southmost extent)
    (E, N, _) = spatial.latlon2utm(-80, 12 - 1e-10)
    assert N == approx(1116915.04)


def test_utm2latlon_sitecompare(olat, olon):
    # Value compared with MATLAB known good numbers
    (lat, lon) = spatial.utm2latlon(447464.9005, 5888516.6988, zone=32)
    assert (lat, lon) == approx((olat, olon))


def test_utm2latlon_failure(olat, olon):
    earray = np.array([447464.9005, 447464.9005, 447464.9005])
    narray = np.array([5888516.6988, 5888516.6988, 5888516.6988])
    with raises(ValueError):
        spatial.utm2latlon(earray, narray[0:2], zone=32)
    with raises(ValueError):
        spatial.utm2latlon(earray[0:2], narray, zone=32)


def test_utm2latlon_arrayin(olat, olon):
    # Value compared with MATLAB known good numbers
    earray = np.array([447464.9005, 447464.9005, 447464.9005])
    narray = np.array([5888516.6988, 5888516.6988, 5888516.6988])
    latlonlist = spatial.utm2latlon(earray, narray, zone=32)
    for latlon in latlonlist:
        assert(latlon[0], latlon[1]) == approx((olat, olon))


def test_utm2latlon_pandas(olat, olon):
    # Value compared with MATLAB known good numbers
    earray = np.array([447464.9005, 447464.9005, 447464.9005])
    narray = np.array([5888516.6988, 5888516.6988, 5888516.6988])
    ind = [1, 2, 3]
    eseries = pd.Series(earray, index=ind)
    nseries = pd.Series(narray, index=ind)
    latlonlist = spatial.utm2latlon(eseries, nseries, zone=32)
    assert isinstance(latlonlist, pd.DataFrame)
    assert (latlonlist.columns == ['lat', 'lon']).all()
    assert (latlonlist.index == ind).all()
    for latlon in np.array(latlonlist):
        assert (latlon[0], latlon[1]) == approx((olat, olon))


def test_utm2latlon_theory():
    # Test theoretically known point for zone 32
    (lat, lon) = spatial.utm2latlon(500000, 0, zone=32)
    assert (lat, lon) == approx((0, 9))

    # Test theoretically known point for zone 1
    (lat, lon) = spatial.utm2latlon(500000, 0, zone=1)
    assert (lat, lon) == approx((0, -177))


def test_utm2latlon_32north_defn():
    # Coords from https://epsg.io/32632

    # Bottom Left Corner (westmost extent)
    (lat, lon) = spatial.utm2latlon(166021.44, 0, 32)
    assert (lat, lon) == approx((0, 6))

    # Center
    (lat, lon) = spatial.utm2latlon(500000, 4649776.22, 32)
    assert (lat, lon) == approx((84/2, 9))

    # Top Right Corner (also northmost extent)
    (lat, lon) = spatial.utm2latlon(534994.66, 9329005.18, 32)
    assert (lat, lon) == approx((84, 12 - 1e-10))


def test_utm2latlon_32south_defn():
    # Test for the southern hemisphere
    # Coords from https://epsg.io/32732

    # Bottom Left Corner
    (lat, lon) = spatial.utm2latlon(441867.78, 1116915.04, 32, True)
    assert (lat, lon) == approx((-80, 6))

    # Center
    (lat, lon) = spatial.utm2latlon(500000, 5572242.78, 32, True)
    assert (lat, lon) == approx((-80/2, 9))

    # Top Right Corner
    (lat, lon) = spatial.utm2latlon(833978.56, 10000000.00, 32, True)
    assert (lat, lon) == approx((0 - 1e-12, 12 - 1e-10))


def test_latlon2lcs_origin(olat, olon):
    (E, N) = spatial.latlon2lcs(olat, olon, olat, olon)
    assert (E, N) == approx((0, 0))


def test_latlon2lcs_shift(olat, olon, dlat, dlon):
    (E, N) = spatial.latlon2lcs(olat + dlat, olon + dlon, olat, olon)
    assert 39 < E < 41
    assert 39 < N < 41

    (E, N) = spatial.latlon2lcs(olat - dlat, olon - dlon, olat, olon)
    assert -41 < E < -39
    assert -41 < N < -39

    (E, N) = spatial.latlon2lcs(olat + dlat, olon + dlon, olat, olon, zone=32)
    assert 39 < E < 41
    assert 39 < N < 41

    (E, N) = spatial.latlon2lcs(olat - dlat, olon - dlon, olat, olon, zone=32)
    assert -41 < E < -39
    assert -41 < N < -39


def test_lcs2latlon_origin(olat, olon):
    lat, lon = spatial.lcs2latlon(0, 0, olat, olon)
    assert (lat, lon) == approx((olat, olon))


def test_lcs2latlon_shift(olat, olon, dlat, dlon):
    lat, lon = spatial.lcs2latlon(40, 40, olat, olon)
    assert (lat, lon) == approx((olat + dlat, olon + dlon), rel=1e-5)

    lat, lon = spatial.lcs2latlon(-40, -40, olat, olon)
    assert (lat, lon) == approx((olat - dlat, olon - dlon), rel=1e-5)

    # Repeat with specified zone
    lat, lon = spatial.lcs2latlon(40, 40, olat, olon, 32)
    assert (lat, lon) == approx((olat + dlat, olon + dlon), rel=1e-5)

    lat, lon = spatial.lcs2latlon(-40, -40, olat, olon, 32)
    assert (lat, lon) == approx((olat - dlat, olon - dlon), rel=1e-5)


def test_dot_list():
    dp = spatial.dot([1, 1], [-1, 1])
    assert dp == approx(0)

    dp = spatial.dot([1, 1], [1, 1])
    assert dp == approx(2)


def test_dot_numpy():
    dp = spatial.dot(np.array([1, 1]), np.array([-1, 1]))
    assert dp == approx(0)

    dp = spatial.dot(np.array([1, 1]), np.array([1, 1]))
    assert dp == approx(2)


def test_unit_list():
    u1 = spatial.unit([1, 0])
    assert u1 == approx(np.array([1, 0]))
    u2 = spatial.unit([-1, 1])
    assert u2 == approx(np.array([-1/np.sqrt(2), 1/np.sqrt(2)]))


def test_unit_numpy():
    u1 = spatial.unit(np.array([1, 0]))
    assert u1 == approx(np.array([1, 0]))
    u2 = spatial.unit(np.array([-1, 1]))
    assert u2 == approx(np.array([-1/np.sqrt(2), 1/np.sqrt(2)]))


def test_magnitude():
    assert spatial.magnitude([1, 0]) == approx(1)
    assert spatial.magnitude([1, 1]) == approx(np.sqrt(2))
    assert spatial.magnitude(np.array([1, 1])) == approx(np.sqrt(2))


def test_rect2pol():
    v1 = spatial.rect2pol(1, 0)
    assert v1 == approx(np.array([1, 0]))

    v1 = spatial.rect2pol(0, 1)
    assert v1 == approx(np.array([1, np.pi / 2]))

    v1 = spatial.rect2pol(1, 1)
    assert v1 == approx(np.array([np.sqrt(2), np.pi / 4]))

    v1 = spatial.rect2pol(1, -1)
    assert v1 == approx(np.array([np.sqrt(2), -np.pi / 4]))


def test_pol2rect():
    v1 = spatial.pol2rect(1, 0)
    assert v1 == approx(np.array([1, 0]))

    v1 = spatial.pol2rect(1, np.pi / 2)
    assert v1 == approx(np.array([0, 1]))

    v1 = spatial.pol2rect(np.sqrt(2), np.pi / 4)
    assert v1 == approx(np.array([1, 1]))

    v1 = spatial.pol2rect(np.sqrt(2), -np.pi / 4)
    assert v1 == approx(np.array([1, -1]))


def test_rotate_vector_list():
    r1 = spatial.rotate_vector([1, 0], np.deg2rad(90))
    assert r1 == approx([0, 1])

    r1 = spatial.rotate_vector([1, 0], np.deg2rad(-90))
    assert r1 == approx([0, -1])

    r1 = spatial.rotate_vector([1, 1], np.deg2rad(90))
    assert r1 == approx([-1, 1])

    r1 = spatial.rotate_vector([1, 1], np.deg2rad(-90))
    assert r1 == approx([1, -1])

    r1 = spatial.rotate_vector([1, 1], np.deg2rad(45))
    assert r1 == approx([0, np.sqrt(2)])


def test_rotate_vector_numpy():
    r1 = spatial.rotate_vector(np.array([1, 0]), np.deg2rad(90))
    assert r1 == approx([0, 1])

    r1 = spatial.rotate_vector(np.array([1, 0]), np.deg2rad(-90))
    assert r1 == approx([0, -1])

    r1 = spatial.rotate_vector(np.array([1, 1]), np.deg2rad(90))
    assert r1 == approx([-1, 1])

    r1 = spatial.rotate_vector(np.array([1, 1]), np.deg2rad(-90))
    assert r1 == approx([1, -1])

    r1 = spatial.rotate_vector(np.array([1, 1]), np.deg2rad(45))
    assert r1 == approx([0, np.sqrt(2)])


def test_project_points(projected_dxdy, projected_dists):
    vecs = projected_dists.keys()
    for vec in vecs:
        dists = spatial.project_vectors(projected_dxdy, vec)
        assert dists == approx(projected_dists[vec])


def test_project_points_pandas(projected_dxdy, projected_dists):
    vecs = projected_dists.keys()
    for vec in vecs:
        ind = np.arange(len(projected_dxdy))
        frame = pd.DataFrame(projected_dxdy, index=ind)
        dists = spatial.project_vectors(frame, vec)
        assert isinstance(dists, pd.DataFrame)
        assert (dists.columns == ['dist']).all()
        assert (dists.index == ind).all()
        assert np.array(dists) == approx(projected_dists[vec])


def test_compute_vectors(xypts, projected_dxdy):
    xs = xypts[0]
    ys = xypts[1]
    refpt = (xs[0], ys[0])
    vecs = spatial.compute_vectors(xs, ys, refpt)
    for veca, vecb in zip(vecs, projected_dxdy):
        assert veca == approx(vecb)


def test_compute_vectors_pandas(xypts, projected_dxdy):
    xs = xypts[0]
    ys = xypts[1]
    refpt = (xs[0], ys[0])
    ind = np.arange(len(xs))
    vecs = spatial.compute_vectors(pd.Series(xs, index=ind),
                                   pd.Series(ys, index=ind),
                                   refpt)
    assert isinstance(vecs, pd.DataFrame)
    assert (vecs.columns == ['dx', 'dy']).all()
    assert (vecs.index == ind).all()
    for veca, vecb in zip(np.array(vecs), projected_dxdy):
        assert veca == approx(vecb)


@fixture(params=range(9))
def data(request):
    # [   (A),     (B),       (C)]
    # [(Ax, Ay), (Bx, By)), (Cx, Cy)]
    srcdata = [
        [(1, 0), (0, 1),
         (1, 1)],
        # initial rotated -45 degrees
        [(1/np.sqrt(2), 1/np.sqrt(2)), (1/np.sqrt(2), -1/np.sqrt(2)),
         (np.sqrt(2), 0)],
        # initial rotated +135 degrees
        [(-1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
         (-np.sqrt(2), 0)],
        # initial case rotated -60 degrees
        [(np.sqrt(3)/2, 1/2), (1/2, -np.sqrt(3)/2),
         (np.sqrt(2)*np.cos(15*np.pi/180), -np.sqrt(2)*np.sin(15*np.pi/180))],
        # Different lengths - Longer X
        [(4, 0), (0, 1),
         (4, 1)],
        # Longer Y
        [(1, 0), (0, 4),
         (1, 4)],
        # Longer X & Y
        [(3, 0), (0, 4),
         (3, 4)],
        # 45 degrees separation
        [(np.sqrt(2), 0), (1/np.sqrt(2), 1/np.sqrt(2)),
         (np.sqrt(2), 0)],
        # 60 degree separation evenly about x axis
        [(np.sqrt(3), 1), (np.sqrt(3), -1),
         (4/np.sqrt(3), 0)],
    ]
    return srcdata[request.param]


@fixture(params=[1, 10])
def n(request):
    return request.param


def test_compute_intersection(data, n):

    A = data[0]
    B = data[1]
    expected = data[2]

    A = np.repeat(np.array([A]), n, axis=0)
    B = np.repeat(np.array([B]), n, axis=0)
    expected = np.repeat(np.array([expected]), n, axis=0)

    C = spatial.compute_intersection(A, B)

    assert C == approx(expected)
