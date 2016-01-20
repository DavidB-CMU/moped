################################################################################
#                                                                              
# libutils.pyx: C Extensions to Utils python module
#
# Alvaro Collet
# acollet@cs.cmu.edu
#
################################################################################
""" libutils.pyx: C Extensions to Utils python module
    
    List of functions:
    C_find
    C_ProjectPts
    """



# ------------------------------------------------------------------------ # 
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double theta)

@cython.boundscheck(False)
@cython.wraparound(False)
def C_find(np.ndarray[np.uint8_t, cast=True] arr):
    """C_find - C-extension to find. Only works for np.uint8 arrays.
    
    Usage: idx = C_find(arr.astype(np.uint8))

    Input:
        arr - N-by-1 Input condition array

    Output:
        idx - M-by-1 output array with indexes for each value that
              arr[i] == True
    """

    cdef int i, counter = 0
    cdef np.ndarray[np.int_t] out = np.zeros((arr.size,), dtype=np.int)
    for i in range(arr.size):
        if arr[i]:
            out[counter] = i
            counter += 1
    out.resize((counter,))
    return out

# ------------------------------------------------------------------------ # 
@cython.boundscheck(False)
@cython.wraparound(False)
def C_ProjectPts(np.ndarray[np.float64_t, ndim=2] pts3D,
                 np.ndarray[np.float64_t, ndim=2] cam_pose,
                 np.ndarray[np.float64_t, ndim=1] KK, 
                 int filter_negZ=True):
    """ C_ProjectPts - Project 3D points to camera [C_Extension]

    Usage: pts2D = ProjectPts(pts3D, cam_pose, KK, [filter_negZ])

    Input:
        pts3D - 3-by-N numpy array of points
        cam_pose - 3-by-4 numpy array of camera extrinsic
                   parameters (world-to-image)
        KK - Camera Intrinsic parameters [fx fy cx cy]
        filter_negZ{True} - If true, filters all points behind the camera
                            (Z < 0) and sets them to [nan nan].

    Output:
        pts2D - 2-by-N numpy array of points [x y] in pixel coords as 
                projected in the image plane of CalibImage.
    """
    cdef np.ndarray[np.float64_t, ndim=2] pts2D = \
        np.zeros((2, pts3D.shape[1]), dtype=np.float64)

    cdef long int i
    cdef double X, Y, Z

    for i in range(pts3D.shape[1]):
        X = cam_pose[0,0]*pts3D[0,i] + cam_pose[0,1]*pts3D[1,i] + \
            cam_pose[0,2]*pts3D[2,i] + cam_pose[0,3]
        Y = cam_pose[1,0]*pts3D[0,i] + cam_pose[1,1]*pts3D[1,i] + \
            cam_pose[1,2]*pts3D[2,i] + cam_pose[1,3]
        Z = cam_pose[2,0]*pts3D[0,i] + cam_pose[2,1]*pts3D[1,i] + \
            cam_pose[2,2]*pts3D[2,i] + cam_pose[2,3]

        pts2D[0, i] = KK[0] * X / Z + KK[2]
        pts2D[1, i] = KK[1] * Y / Z + KK[3]

        if filter_negZ and Z<0:
            pts2D[0, i] = np.nan
            pts2D[1, i] = np.nan

    return pts2D

# ------------------------------------------------------------------------ # 
@cython.boundscheck(False)
@cython.wraparound(False)
def C_ProjectPtsOut(np.ndarray[np.float64_t, ndim=2] pts2D,
                    np.ndarray[np.float64_t, ndim=1] KK, 
                    np.ndarray[np.float64_t, ndim=2] cam_pose):
    """C_ProjectPtsOut - Use inverse projection to map pts in 2D to 3D rays.
    [C-Extension]

    Usage: vec3D, cam_center = projectPtsOut(pts2D, KK, cam_pose);

    Input:
    pts2D - 2-by-N array of 2D points to be backprojected, np.float64
    KK - internal camera parameters: [fx fy px py], dtype=np.float64 
    cam_pose - 3-by-4 or 4-by-4 matrix with world-to-cam camera params,
               dtype=np.float64
    
    Output:
    vec3D - 3-by-N array of 3D vectors backprojected from the camera plane
    cam_center - 3-by-1 point that works as camera center. All rays
        backprojected from the camera pass through cam_center and vec3D, 
        i.e.:  line(lambda) = cam_center + lambda*vec3D;
    """
    cdef np.ndarray[np.float64_t, ndim=2] vec3D = \
        np.ones((3, pts2D.shape[1]), dtype=np.float64)

    cdef long int i
    cdef double X, Y, Z
    cdef np.ndarray[np.float64_t, ndim=1] iT = np.zeros((3,), \
                                                        dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] Kinv

    Kinv = np.array([1/KK[0], 1/KK[1], -KK[2]/KK[0], -KK[3]/KK[1]], \
                    dtype=np.float64)

    # Remember that we're using pose inverse R' = R.T, T' = -R.T*T
    iT = np.dot(-cam_pose[:3,:3].T, cam_pose[:3,3])

    for i in range(pts2D.shape[1]):
        vec3D[0, i] = Kinv[0] * pts2D[0,i] + Kinv[2]
        vec3D[1, i] = Kinv[1] * pts2D[1,i] + Kinv[3]

    # Normalize vectors
    vec3D = vec3D / np.sqrt(np.sum(vec3D**2, axis=0))
 
    # Put vectors in world coords (just rotate them)
    vec3D = np.dot(cam_pose[:3,:3].T, vec3D)

    return vec3D, iT 

# ------------------------------------------------------------------------ #
@cython.boundscheck(False)
@cython.wraparound(False)
def C_GraphDistances(np.ndarray[np.float64_t, ndim=2] pts,
                     np.ndarray[np.int_t, ndim=2] edges):
    """C_GraphDistances - Compute distances between connected nodes in 
    graph. [C-Extension]

    Usage: distances = C_GraphDistances(pts, edges);

    Input:
    pts - N-by-M array or N-dimensional points.
    edges - K-by-M array of indexes of neighbors, where
            pts[:,i] <--> edges[:,i]. Edges with negative or NaN value
            are considered not valid, and output a distance of NaN.
    
    Output:
    distances - K-by-M array of distances between edges and pts, where
        distances[i,j] = dist_l2(pts[:,i], edges[j,i]).
    """
    cdef int nDim = pts.shape[0], nEdges = edges.shape[0]
    cdef int nPts = pts.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] distances = \
        np.nan * np.ones((nEdges, nPts), dtype=np.float64)

    cdef long int p, k

    for p in range(nPts):
        for k in range(nEdges):
            # If edges < 0, no edge
            if edges[k,p] >= 0:
                distances[k, p] = np.linalg.norm(pts[:,p] - \
                                                 pts[:,edges[k,p]])
    
    return distances 

# ------------------------------------------------------------------------ #
@cython.boundscheck(False)
@cython.wraparound(False)
def C_distRay3D(np.ndarray[np.float64_t, ndim=2] pts3D,
                np.ndarray[np.float64_t, ndim=2] rays3D, 
                np.ndarray[np.float64_t, ndim=2] cam_pose):
    """distRay3D - Compute distances from 3D point(s) to 3D ray(s).
    For a ray3D = [v, c], the distance to a point X = RP+t is computed as:
    dist = || (I-vv')(X-c) ||; [C-Extension]

    Usage: dist = distRay3D(pts3D, rays3D, cam_pose);

    Input:
    pts3D - 3-by-N array of 3D points to compute distances from.
    rays3D - 6-by-N array of rays [v(1:3,:); c(4:6,:)] to project pts3D to.
            v is the ray direction (a vector) and c is a point in the
            ray (e.g. the camera center).
    cam_pose - 3-by-4 or 4-by-4 matrix with world-to-cam camera params,
               dtype=np.float64

    Output:
    dist - N-by-1 distance vector. Dist(i) = dist(pts3D(:,i), rays3D(:,i)).

    NOTE: This function computes 1 point to 1 ray distances, NOT all rays
    to all points!
    """    
    cdef np.ndarray[np.float64_t, ndim=2] R = cam_pose[:3,:3]
    cdef np.ndarray[np.float64_t, ndim=1] T = cam_pose[:3,3]
    cdef np.ndarray[np.float64_t, ndim=2] tx_pts3D
    cdef np.ndarray[np.float64_t, ndim=2] Pcentered 
    cdef np.ndarray[np.float64_t, ndim=1] dotp
    cdef np.ndarray[np.float64_t, ndim=2] dist 

    # Transform points according to camera position
    tx_pts3D = np.dot(R, pts3D) + T[:,None]

    # Pcentered --> X-c
    Pcentered = tx_pts3D - rays3D[3:, :]

    # dotp --> v'(X-c)
    dotp = np.sum(rays3D[0:3, :] * Pcentered, axis=0)

    # This is it: dist = (I-vv')(X-c) = (X-c) - vv'(X-c)
    dist = Pcentered - (rays3D[0:3, :] * dotp)
    return np.sqrt(np.sum(dist**2, axis=0))

    # Optional: for points in front of the camera, the magnitude of the
    # dot product (1-dotp) should be greater than zero
    #mag_dotp = 1 - dotp / np.sqrt(np.sum(Pcentered**2)); # mag_dotp >= 0
    #mag_dotp = np.max(mag_dotp - 0.1, axis=0); # If 1-dotp < 0.1, consider it 0

# ------------------------------------------------------------------------ #
@cython.boundscheck(False)
@cython.wraparound(False)
def C_distRay3D_fast(np.ndarray[np.float64_t, ndim=2] pts3D,
                     np.ndarray[np.float64_t, ndim=1] vec3D, 
                     np.ndarray[np.float64_t, ndim=1] c, 
                     np.ndarray[np.float64_t, ndim=2] cam_pose):
    """distRay3D - Compute distances from 3D point(s) to 3D ray(s).
    For a ray3D = [v, c], the distance to a point X = RP+t is computed as:
    dist = || (I-vv')(X-c) ||; [C-Extension]

    Usage: dist = distRay3D(pts3D, vec3D, c, cam_pose);

    Input:
    pts3D - 3-by-N array of 3D points to compute distances from.
    vec3D - 3-by-1 np.float64 ray (3D vector) to project pts3D to.
            v is the ray direction (a vector) and c is a point in the
            ray (e.g. the camera center).
    c - 3-by-1 np.float64 point in the ray, so that ray = (vec3D, c)
    cam_pose - 3-by-4 or 4-by-4 matrix with world-to-cam camera params,
               dtype=np.float64

    Output:
    dist - N-by-1 distance vector. Dist(i) = dist(pts3D(:,i), ray).
    argmin - Index to minimum distance

    NOTE: This function computes all points to 1 ray distances.
    """    
    cdef np.ndarray[np.float64_t, ndim=1] dist = \
            np.zeros((pts3D.shape[1],), dtype=np.float64)
    cdef int i, argmin = 0
    cdef double X, Y, Z, Xc, Yc, Zc, dX, dY, dZ, dotp, val = 100000
    

    # Transform points according to camera position
    for i in range(pts3D.shape[1]):
        X = cam_pose[0,0]*pts3D[0,i] + cam_pose[0,1]*pts3D[1,i] + \
            cam_pose[0,2]*pts3D[2,i] + cam_pose[0,3]
        Y = cam_pose[1,0]*pts3D[0,i] + cam_pose[1,1]*pts3D[1,i] + \
            cam_pose[1,2]*pts3D[2,i] + cam_pose[1,3]
        Z = cam_pose[2,0]*pts3D[0,i] + cam_pose[2,1]*pts3D[1,i] + \
            cam_pose[2,2]*pts3D[2,i] + cam_pose[2,3]

        # XYZc = XYZ - c
        Xc = X - c[0]
        Yc = Y - c[1]
        Zc = Z - c[2]

        # dotp = v' * XYZc
        dotp = Xc*vec3D[0] + Yc*vec3D[1] + Zc*vec3D[2]
        
        # dist = (I-vv')(X-c) = XYZc - v * v' * XYZc
        dX = Xc - vec3D[0] * dotp
        dY = Yc - vec3D[1] * dotp
        dZ = Zc - vec3D[2] * dotp

        dist[i] = sqrt(dX*dX + dY*dY + dZ*dZ)
        
        if dist[i] < val:
            argmin = i
            val = dist[i]
    return dist, argmin

# ------------------------------------------------------------------------ #
@cython.boundscheck(False)
def C_sub2ind(np.ndarray[np.int_t, ndim=2] idx, tuple dims):
    """C_sub2ind - From ND-indexes, compute their flattened 1d equivalent
        indexes (equivalent to sub2ind in Matlab). This is a fast
        extension compiled in cython.

        Usage: flat_idx = C_sub2ind(idx, dims)

        Input:
            idx - N-by-M array of N-dimensional indexes
            dims - N-tuple containing indexed array dimensions

        Output:
            flat_idx - N-by-1 array containing flat indexes, such that
                       array[idx[0,:], idx[1,:]..., idx[N,:]] = \
                       array.flat[flat_idx]
    """
    cdef np.ndarray[np.int_t, ndim=1] offsets

    offsets = np.cumprod(dims[1:][::-1])[::-1]
    return (np.sum(idx[:-1,:]*offsets[:,None], axis=0) + idx[-1,:])

# ------------------------------------------------------------------------ #
@cython.boundscheck(False)
def C_ind2sub(np.ndarray[np.int_t, ndim=1] flat_idx, tuple dims):
    """C_ind2sub - From flat indexes, compute their ND-equivalent
        indexes (equivalent to ind2sub in Matlab). This function is
        equivalent to numpy.unravel_index, but works with arrays of indexes.

        Usage: idx = C_ind2sub(flat_idx, dims)

        Input:
            flat_idx - N-by-1 array containing flat indexes, such that
                array[idx[:,0], idx[:,1]..., idx[:,N]] = \
                array.flat[flat_idx]
            dims - M-tuple containing indexed array dimensions

        Output:
            idx - N-by-M array of N-dimensional indexes
    """
    cdef np.ndarray[np.int_t, ndim=2] idx = \
            np.zeros((len(dims), flat_idx.size), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] offsets = \
            np.cumprod(dims[1:][::-1])[::-1]
    cdef int i, off

    for i in range(offsets.size):
        idx[i,:] = flat_idx / offsets[i]
        flat_idx = np.remainder(flat_idx, offsets[i]) 

    idx[-1,:] = flat_idx
    return idx


