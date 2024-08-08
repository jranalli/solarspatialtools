import numpy as np
import pandas as pd
from pyproj import Proj


def _calc_zone(lon):
    """
    Calculate the UTM zone for a given longitude, based on a 6deg window model.

    Parameters
    ----------
    lon : numeric
        The longitude in degrees

    Returns
    -------
    zone : numeric
        The UTM zone (number only, alphabetic N/S indicator excluded)
    """
    z = int((lon-(-180))/6+1)
    # minimum command accounts for +180 which otherwise produces zone 61.
    return min(z, 60)


def latlon2utm(lat, lon, zone=None):
    """
    Convert a latitude/longitude pair from WGS84 into UTM coordinates.

    Parameters
    ----------

    lat : numeric
        a value of latitude in degrees, or an indexable array of latitudes,
        or a Pandas Series (indices for lat and lon must match)

    lon : numeric
        a value of latitude in degrees, or an indexable array of latitudes,
        or a Pandas Series (indices for lat and lon must match)

    zone : numeric
        the UTM zone (number only). Calculated from longitude if None

    Returns
    -------
    (east, north) : numeric
        a tuple of the UTM (Easting, Northing) for the site. For input arrays
        or lists, the output will be a numpy array of E,N pairs. For inputs
        that are Pandas types, the output will have columns 'E' and 'N' and
        will share an index with the input.

    """

    if hasattr(lat, "__len__"):
        if not len(lat) == len(lon):
            raise ValueError("lat and lon arrays must have the same length.")
        if hasattr(lat, "index") \
                and isinstance(lat, (pd.DataFrame, pd.Series)):
            data = np.array([latlon2utm(lati, loni, zone)
                             for (lati, loni) in zip(lat, lon)])
            return pd.DataFrame(data, index=lat.index,
                                columns=["E", "N", "zone"])
        else:
            return np.array([latlon2utm(lati, loni, zone)
                             for (lati, loni) in zip(lat, lon)])
    else:
        if zone is None:
            zone = _calc_zone(lon)

        south = (lat < 0)

        p1 = Proj(proj='utm', zone=zone, ellipsis='WGS84', datum='WGS84',
                  units='m', south=south, preserve_units=True)
        (E, N) = p1(lon, lat)
        return E, N, zone


def utm2latlon(e, n, zone, south=False):
    """
    Convert from UTM coordinates to latitude/longitude (WGS84)

    Parameters
    ----------
    e : numeric
        a value of easting, or an indexable array of eastings, or a Pandas
        Series (indices for e and n must match)

    n : numeric
        a value of northing, or an indexable array of northings, or a Pandas
        Series (indices for e and n must match)

    zone : numeric
        the UTM zone, which is required for this conversion. Zone should
        include only the number (i.e. 32U should be listed as 32).

    south : bool
        False for northern hemisphere, True for southern hemisphere

    Returns
    -------
    (lat, lon) : numeric
        tuple containing latitude, longitude in degrees. For input arrays
        or lists, the output will be a numpy array of lat,lon pairs. For inputs
        that are Pandas types, the output will have columns 'lat' and 'lon' and
        will share an index with the input.
    """

    if hasattr(e, "__len__"):
        if not len(e) == len(n):
            raise ValueError("e and n arrays must have the same length.")
        if hasattr(e, "index") \
                and isinstance(e, (pd.DataFrame, pd.Series)):
            data = [utm2latlon(ei, ni, zone, south) for (ei, ni) in zip(e, n)]
            return pd.DataFrame(data, index=e.index, columns=['lat', 'lon'])
        else:
            return [utm2latlon(ei, ni, zone, south) for (ei, ni) in zip(e, n)]
    else:
        p1 = Proj(proj='utm', zone=zone, ellipsis='WGS84', datum='WGS84',
                  units='m', south=south)
        (lon, lat) = p1(e, n, inverse=True)
        return lat, lon


def latlon2lcs(lat, lon, origin_lat, origin_lon, zone=None):
    """
    Convert from lat/lon to a referenced local coordinate system (LCS) for
    wobas. LCS uses a UTM projection but offsets the values to a given origin.

    Parameters
    ----------
    lat : numeric
        target point latitude in deg

    lon : numeric
        target point longitude in deg

    origin_lat : numeric
        The latitude of the LCS origin in degrees

    origin_lon : numeric
        The longitude of the LCS origin in degrees

    zone : numeric
        The UTM zone (obtained from origin longitude if not specified)

    Returns
    -------
    (east, north) : numeric
        The Easting and Northing of the target point in the LCS coordinates
    """
    # Convert both coordinates to UTM
    e_i, n_o, _ = latlon2utm(origin_lat, origin_lon, zone=zone)
    e_t, n_t, _ = latlon2utm(lat, lon, zone=zone)

    # Shift by origin to convert to LCS
    # (LCS is UTM, re-zeroed to the origin's coordinates)
    e_t -= e_i
    n_t -= n_o

    return e_t, n_t


def lcs2latlon(east, north, origin_lat, origin_lon, zone=None):
    """
    Convert from the origin referenced local coordinate system (LCS) for wobas
    back to a lat/lon coordinate system. LCS uses a UTM projection but offsets
    the values to a given origin.
    Parameters
    ----------
    east : numeric
        The easting of the point in the LCS

    north : numeric
        The northing of the point in the LCS

    origin_lat : numeric
        The latitude of the LCS origin in degrees

    origin_lon : numeric
        The longitude of the LCS origin in degrees

    zone : numeric
        The UTM zone (obtained from origin longitude if not specified)

    Returns
    -------
    (lat, lon) : numeric
        Tuple containing the latitude and longitude of the point in degrees
    """
    # Calculate the zone and south flag
    if zone is None:
        zone = _calc_zone(origin_lon)
    south = origin_lat < 0

    # Get the origin in UTM
    e_o, n_o, _ = latlon2utm(origin_lat, origin_lon)

    # Shift the target
    e_t = e_o + east
    n_t = n_o + north

    # Calculate
    lat, lon = utm2latlon(e_t, n_t, zone, south)
    return lat, lon


def dot(vec_a, vec_b):
    """
    Vector dot product for 2-D cartesian

    Parameters
    ----------
    vec_a : numeric
        the 2-D cartesian vector as an (x, y) indexable

    vec_b : numeric
        the 2-D cartesian vector as an (x, y) indexable

    Returns
    -------
    A dot B : numeric
        the dot product of the two vectors
    """
    return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]


def unit(vec):
    """
    Get the unit vector matching a vector's direction

    Parameters
    ----------
    vec : numeric
        the 2-D cartesian vector as an (x, y) indexable

    Returns
    -------
    (xi, yi) : numeric
        the 2-D cartesian vector scaled to a magnitude of 1
    """
    return vec / magnitude(vec)


def magnitude(vec):
    """
    Get the magnitude of a vector

    Parameters
    ----------
    vec : numeric
        the 2-D cartesian vector as an (x, y) indexable

    Returns
    -------
    magnitude : numeric
        the magnitude of the vector
    """
    return np.sqrt(vec[0]**2+vec[1]**2)


def rect2pol(x, y):
    """
    2D cartesian vector to polar form

    Parameters
    ----------
    x : numeric
        the x position

    y : numeric
        the y position

    Returns
    -------
    (r, theta) : numeric
        the polar form (r, theta) with theta in +/- pi radians
    """
    r = magnitude([x, y])
    theta = np.arctan2(y, x)
    return r, theta


def pol2rect(r, theta):
    """
    2D polar vector to cartesian form

    Parameters
    ----------
    r : numeric
        the radius coordinate

    theta : numeric
        the angle coordinate in radians

    Returns
    -------
    (x, y) : numeric
        a tuple of the vector
    """

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def rotate_vector(vector, theta):
    """
    2D rotation of a vector in cartesian form

    Parameters
    ----------
    vector : (x, y) numeric
        A tuple (or numpy array) containing the input vector

    theta : numeric
        Angle of rotation in radians

    Returns
    -------
    (x, y) : numeric
        A tuple (or numpy array) containing the rotated vector. Matches data
        type of input.
    """

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    newvec = np.matmul(rot_matrix, vector)

    if type(vector) in [type(()), type([])]:
        return tuple(newvec)
    else:
        return newvec


def project_vectors(vectors, dir_vector):
    """
    Convert a set of vectors into a list of distances in a given direction.
    See also: compute_vectors for computing the vector array. Units will match
    those provided for the input vectors. Should probably be done in a
    rectilinear system like UTM.

    Parameters
    ----------
    vectors : numeric
        iterable set of vectors pointing from a reference to sites of interest.

    dir_vector : numeric
        tuple of (x, y) showing the cartesian direction that the vectors should
        be projected onto. Length of vector greater than 1 will be ignored
        (i.e. will be converted to unit vector as the projection is done).

    Returns
    -------
    dot_positions : numeric
        The dot product of each vector with the projection vector. Essentially
        represents the length of each vector in the given directions, with
        negative numbers implying opposite pointing. If the input is a Pandas
        type, the output will be in a DataFrame with an index that matches the
        input and has columns 'dist'.
    """

    if hasattr(vectors, "index") \
            and isinstance(vectors, (pd.DataFrame, pd.Series)):
        pandas_mode = True
        vec_array = np.array(vectors)
    else:
        pandas_mode = False
        vec_array = vectors

    unit_dir = unit(dir_vector)
    v = np.array(vec_array)
    dot_positions = v[:, 0] * unit_dir[0] + v[:, 1] * unit_dir[1]

    if pandas_mode:
        return pd.DataFrame(dot_positions,
                            index=vectors.index, columns=['dist'])
    else:
        return dot_positions


def compute_vectors(x, y, refpt):
    """
    Compute a list of vectors pointing from a reference origin to each site.
    Units will match those given by x, y, which means that ideally this
    function would be called with inputs in a rectilinear system such as UTM.

    Parameters
    ----------
    x : numeric
        An indexable array of x coordinates

    y : numeric
        An indexable array of y coordinates

    refpt : numeric
        The reference position as a tuple (x, y) or other indexable pair

    Returns
    -------
    vectors : numeric
        An ordered list of vectors (x,y) from ref to each point in meters. If
        the input is a Pandas type, the output will be in a DataFrame with an
        index that matches the input and has columns 'dx' and 'dy'.

    """

    # Find ref
    pt0 = np.array(refpt)[0:2]  # Make sure it doesn't contain extra columns

    # Calculate each point's vector
    vectors = []
    for x_i, y_i in zip(x, y):
        pti = np.array((x_i, y_i))

        # Compute the vector
        ptdiff = pti - pt0
        vectors.append(ptdiff)

    if hasattr(x, "index") and isinstance(x, (pd.DataFrame, pd.Series)):
        return pd.DataFrame(np.array(vectors),
                            index=x.index, columns=['dx', 'dy'])
    else:
        return np.array(vectors)


def compute_intersection(A, B):
    """
    Computes intersection, C, of the lines perpendicular to the tips of two
    vectors (A and B) who share a starting point at the origin.

    Vectors are specified in nx2 arrays and computations of Ci are made for
    each vector pair Ai, Bi.

    Parameters
    ----------
    A : np.array
        nx2 array of the vectors. Outer index is the vector instance, inner
        index is the coordinate x,y:
        [[ax1,ay1], [ax2,ay2], ..., [axn,ayn]]
    B : np.array
        nx2 array of the vectors. Outer index is the vector instance, inner
        index is the coordinate x,y:
        [[bx1,by1], [bx2,by2], ..., [bxn,byn]]

    Returns
    -------
    C : np.array
        nx2 array of computed positions of the intersection.
        [[cx1,cy1], [cx2,cy2], ..., [cxn,cyn]]
    """

    # Alternate method available via line intersections at:
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    # Derive by computing unit vectors a & b in dir of A&B
    # Then compute a dot C and b dot C. These should be equal to the
    # magnitudes of A and B. You now have an equation for cx, cy in terms
    # of the known a, b, A, B. Thanks @ Nate Williams and David Starling.

    Ax = A[:, 0]
    Ay = A[:, 1]
    Bx = B[:, 0]
    By = B[:, 1]
    Amag2 = Ax**2 + Ay**2  # magnitude squared
    Bmag2 = Bx**2 + By**2
    cx = (Bmag2 * Ay - By * Amag2) / (Bx*Ay-Ax*By)
    cy = (Amag2 * Bx - Ax * Bmag2) / (Bx*Ay-Ax*By)

    C = np.array([cx, cy]).T

    return C
