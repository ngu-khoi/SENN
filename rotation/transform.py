import cv2
import numpy as np
from typing import NamedTuple, Optional, Tuple


class ImageSize(NamedTuple):
    height: int
    width: int


def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Transform points using a given transformation matrix.

    Args:
        transform_matrix (np.ndarray): 3x3 transformation matrix.
        points (np.ndarray): Nx2 array of points.

    Returns:
        np.ndarray: Nx2 array of transformed points.
    """
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T
    points = transform_matrix.dot(points)
    points = np.true_divide(points, points[-1])
    points = points[:2].T
    return points


def get_transform_matrix(
    image_size: ImageSize,
    x_rotation: float,
    y_rotation: float,
    z_rotation: float,
    x_translate: float,
    y_translate: float,
    focal_length: Optional[float] = None,
) -> Tuple[np.ndarray, ImageSize]:
    """Calculate 3x3 transformation matrix for a given set of operations.

    Args:
        image_size (Tuple[int, int]): Original Image size (height, width).
        x_rotation (float): Rotation along X (Horizontal) axis.
        y_rotation (float): Rotation along Y (Vertical) axis.
        z_rotation (float): Rotation along Z (Inward) axis.
        x_translate (float): Relative Translation along X (Horizontal) axis.
        y_translate (float): Relative Translation along Y (Vertical) axis.
        focal_length (Optional[float], optional): Translation along Z (Inward) axis
            which should be considered as focal length. If set to None then it will be
            calculated automaticly. Defaults to None.

    Returns:
        Tuple[np.ndarray, ImageSize]: A tuple of transformation matrix and new image
            size.
    """
    x_rotation, y_rotation, z_rotation = map(
        np.deg2rad,
        (x_rotation, y_rotation, z_rotation),
    )

    height, width = image_size

    # calculating focal length
    if focal_length is None:
        focal_length = np.sqrt(height**2 + width**2)
        if np.sin(z_rotation) != 0:
            focal_length /= 2 * np.sin(z_rotation)

    z_translate = focal_length

    projection_2d_to_3d = np.array(
        [
            [1, 0, -width / 2],
            [0, 1, -height / 2],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    rotation_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x_rotation), -np.sin(x_rotation), 0],
            [0, np.sin(x_rotation), np.cos(x_rotation), 0],
            [0, 0, 0, 1],
        ]
    )

    rotation_y = np.array(
        [
            [np.cos(-y_rotation), 0, -np.sin(-y_rotation), 0],
            [0, 1, 0, 0],
            [np.sin(-y_rotation), 0, np.cos(-y_rotation), 0],
            [0, 0, 0, 1],
        ]
    )

    rotation_z = np.array(
        [
            [np.cos(z_rotation), -np.sin(z_rotation), 0, 0],
            [np.sin(z_rotation), np.cos(z_rotation), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    rotation_matrix = (rotation_x @ rotation_y) @ rotation_z

    translation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z_translate],
            [0, 0, 0, 1],
        ]
    )

    projection_3d_to_2d = np.array(
        [
            [focal_length, 0, width / 2, 0],
            [0, focal_length, height / 2, 0],
            [0, 0, 1, 0],
        ]
    )

    transform_matrix = projection_3d_to_2d @ (
        translation_matrix @ (rotation_matrix @ projection_2d_to_3d)
    )

    corners = np.array(
        [
            [0, 0],
            [0, height],
            [width, height],
            [width, 0],
        ],
        dtype=np.float32,
    )

    # Fix translation issue
    corners = transform_points(corners, transform_matrix)
    xmin, ymin = map(int, corners.min(axis=0))
    xmax, ymax = map(int, corners.max(axis=0))
    new_h = ymax - ymin
    new_w = xmax - xmin

    translate = np.eye(3)

    # Convert XY translations to absolute coordinates
    x_translate *= new_w
    y_translate *= new_h
    translate[0, 2] = -xmin + x_translate
    translate[1, 2] = -ymin + y_translate
    transform_matrix = translate @ transform_matrix

    return transform_matrix, ImageSize(new_h, new_w)


def transform_image(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    after_transform_image_size: ImageSize,
    cv2_warp_perspective_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Transform an image using a given transformation matrix.

    Args:
        image (np.ndarray): Image to be transformed.
        transform_matrix (np.ndarray): Transformation matrix.
        after_transform_image_size (ImageSize): Size of the image after transformation.
        cv2_warp_perspective_kwargs (Optional[dict], optional): Additional arguments
            to be passed to cv2.warpPerspective. Defaults to None.

    Returns:
        np.ndarray: Transformed image.
    """
    kwargs = dict(
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0],
        flags=cv2.INTER_CUBIC,
    )
    if cv2_warp_perspective_kwargs is not None:
        kwargs.update(cv2_warp_perspective_kwargs)
    new_h, new_w = after_transform_image_size
    warp_result = cv2.warpPerspective(
        image,
        transform_matrix,
        (new_w, new_h),
        **kwargs,
    )
    return warp_result


def rotate_image(
    image: np.ndarray,
    x_rotation: float,
    y_rotation: float,
    z_rotation: float,
    focal_length: Optional[float] = None,
    cv2_warp_perspective_kwargs: Optional[dict] = None,
):
    """Rotate an image.

    Args:
        image (np.ndarray): Image to be rotated.
        x_rotation (float): Rotation angle around the x-axis.
        y_rotation (float): Rotation angle around the y-axis.
        z_rotation (float): Rotation angle around the z-axis.
        focal_length (Optional[float], optional): Focal length. Defaults to None.
        cv2_warp_perspective_kwargs (Optional[dict], optional): Additional arguments
            to be passed to cv2.warpPerspective. Defaults to None.

    Returns:
        np.ndarray: Rotated image.
    """
    transform_matrix, (new_h, new_w) = get_transform_matrix(
        image.shape[:2],
        x_rotation,
        y_rotation,
        z_rotation,
        x_translate=0,
        y_translate=0,
        focal_length=focal_length,
    )
    return transform_image(
        image,
        transform_matrix,
        ImageSize(new_h, new_w),
        cv2_warp_perspective_kwargs=cv2_warp_perspective_kwargs,
    )


def rotate_resize_and_cast(image, x_rotation=0, y_rotation=0, z_rotation=0):
    """Rotates the image, resizes if needed, and casts onto a 28x28 canvas."""
    # Rotate the image
    rotated_img = rotate_image(
        image, x_rotation=x_rotation, y_rotation=y_rotation, z_rotation=z_rotation
    )

    # Check if rotation increased the image dimensions beyond 28x28
    if rotated_img.shape[0] > 28 or rotated_img.shape[1] > 28:
        # Compute scaling factor to fit within 28x28
        scale_factor = 28 / max(rotated_img.shape[:2])
        new_size = (
            max(1, int(rotated_img.shape[1] * scale_factor)),
            max(1, int(rotated_img.shape[0] * scale_factor)),
        )

        # Resize the rotated image to fit within 28x28
        resized_img = cv2.resize(rotated_img, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized_img = rotated_img  # No resizing needed if within bounds

    # Create a blank 28x28 canvas and center the resized image on it
    canvas = np.zeros((28, 28, 3), dtype=np.uint8)
    y_offset = (28 - resized_img.shape[0]) // 2
    x_offset = (28 - resized_img.shape[1]) // 2
    canvas[
        y_offset : y_offset + resized_img.shape[0],
        x_offset : x_offset + resized_img.shape[1],
    ] = resized_img

    return rotated_img, canvas
