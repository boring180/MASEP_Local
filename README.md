# MASEP_Local

## 1.Background

This is a Final Year Project for the Hong Kong University of Science and Technology.

## 2.Overview

## 3.Features

The internal and external callibration only accept images with padding

# Camera calibration

### Distortion

Two types of distortion: radial and tangential

Add up to five parameters

### Intrinsic parameters

$$
camera~matrix = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix}
$$

Where $f_x$ $f_y$  are focal length,  $c_x$ $c_y$ are optical centers

### Extrinsic parameters

$$
translation~matrix = \begin{bmatrix} r_1& r_2& r_3& x \\ r_4& r_5& r_6& y \\ r_7& r_8& r_9& z\\ 0& 0& 0& 1\end{bmatrix}
$$

### Calibration

When calibration, it model a pinhole camera to perform association between object points(3D world) and image points(2D image)

The chessboard will be assumed as stationary, which the object points will be defined as an np array with length of number of internal corners

```python
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,9,0)
objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
objp = objp * square_size
```

The cv function will find the coordinates of the internal corners on the image plane

```python
ret, corners = cv2.findChessboardCorners(gray, (number_of_internal_corners_x,number_of_internal_corners_y), None)
# Corners: numpy array of shape (number_of_internal_corners, 1, 2)
```

With cornerSubPix, the results can be more accurate

```python
corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
```