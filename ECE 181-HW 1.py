import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    from io import BytesIO
    from urllib.request import Request, urlopen

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    return BytesIO, Image, Request, mo, np, plt, urlopen


@app.cell
def _(BytesIO, Image, Request, np, urlopen):
    def image_from_url(image_url):
        """Download image_url and return it as a grayscale NumPy array."""
        request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return np.asarray(Image.open(BytesIO(response.read())).convert("L"), dtype=float)

    return (image_from_url,)


@app.cell(hide_code=True)
def _(image_from_url):
    mare_image = image_from_url(
        "https://bigdatahealth.ucsb.edu/sites/default/files/styles/logo/"
        "public/logo/img_2289.jpeg"
    )
    mare_side = min(mare_image.shape)
    mare_top = (mare_image.shape[0] - mare_side) // 2
    mare_left = (mare_image.shape[1] - mare_side) // 2
    mare_image = mare_image[
        mare_top : mare_top + mare_side, mare_left : mare_left + mare_side
    ]
    storke_tower_image = image_from_url(
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/48/"
        "Ucsb-storke-tower01.jpg/960px-Ucsb-storke-tower01.jpg"
    )
    kavli_image = image_from_url(
        "https://www.kitp.ucsb.edu/sites/default/files/kitp/kavlisb.jpeg"
    )
    return kavli_image, mare_image, storke_tower_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 1

    ## Editing

    Marimo notebooks are just Python files, where each "cell" is a function. You can either edit here, in [Molab](https://molab.marimo.io/), or directly in Python.

    ### Option 1 - Molab [Recommended]

    This is the most recommended option. Take this notebook (.py file), upload it to Molab, and start working. Once you are finished making your edits, click "Export" and select "Python" to export it as a notebook source.

    ### Option 2 - Local Marimo

    To work locally, first create a python project and place this file `hw1.py` within it. Then, install marimo using your favorite package manager (`uv add marimo`). Finally, launch this notebook by typing in your terminal `uv run marimo edit` and a new tab should open in your browser.

    ### Option 3 - Regular Python [Not Recommended]

    It is theoretically possible to edit the `.py` file directly (although we don't recommend it). If you choose this route, make sure that your Python/Markdown renders properly in Marimo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 1: Image Convolution and Edge Detection

    You are given the following two images $I_1$ and $I_2$, with three convolution kernels $S_x$, $S_y$ and $L$. [.] indicates the origin of each image.

    \[
    I_1 = \begin{bmatrix}
    [1] & 1 & 1 \\
    3 & 3 & 3 \\
    6 & 6 & 6
    \end{bmatrix}
    \qquad
    I_2 = \begin{bmatrix}
    [10] & 10 & 10 \\
    10 & 100 & 100 \\
    10 & 100 & 100
    \end{bmatrix}
    \]

    \[
    S_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & [0] & 2 \\ -1 & 0 & 1\end{bmatrix}
    \qquad
    S_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & [0] & 0 \\ 1 & 2 & 1\end{bmatrix}
    \qquad
    L = \begin{bmatrix}0 & 1 & 0 \\ 1 & [-4] & 1 \\ 0 & 1 & 0\end{bmatrix}
    \]
    """)
    return


@app.cell
def _():
    # EDIT BELOW: transcribe the matrices displayed above into NumPy arrays.
    # Replace the zero defaults with the instructor-provided entries for I_1 and I_2.
    image_1 = [[1, 1, 1], [3, 3, 3], [6, 6, 6]]
    image_2 = [[10, 10, 10], [10, 100, 100], [10, 100, 100]]

    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    return image_1, image_2, laplacian, sobel_x, sobel_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(a) Sobel X and Sobel Y By Hand

    Compute the full linear convolutions \(S_x * I_1\) and \(S_y * I_1\) **by hand**, where \(*\) denotes linear convolution. Mark the origin in each output matrix. Use the two responses to locate the edges, and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 1(a)

    For a full linear convolution, we need to zero pad $I_1$ with $2$ zeroes in each direction, as well as flip the kernels $S_x$ and $S_y$. This means our two computations are actually between (1) $S_{x, \text{flipped}}$ and $I_{1, \text{padded}}$ and (2) $S_{y, \text{flipped}}$ and $I_{1, \text{padded}}$:

    \[
    S_{x} * I_{1} = \begin{bmatrix}
    1 & 0 & -1 \\
    2 & [0] & -2 \\
    1 & 0 & -1
    \end{bmatrix}
    *
    \begin{bmatrix}
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & [1] & 1 & 1 & 0 & 0 \\
    0 & 0 & 3 & 3 & 3 & 0 & 0 \\
    0 & 0 & 6 & 6 & 6 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    =
    \begin{bmatrix}
    -1 & -1 & 0 & 1 & 1 \\
    -5 & [-5] & 0 & 5 & 5 \\
    -13 & -13 & 0 & 13 & 13 \\
    -15 & -15 & 0 & 15 & 15 \\
    -6 & -6 & 0 & 6 & 6
    \end{bmatrix}
    \]

    \[
    S_{y} * I_{1} = \begin{bmatrix}
    1 & 2 & 1 \\
    0 & [0] & 0 \\
    -1 & -2 & -1
    \end{bmatrix}
    *
    \begin{bmatrix}
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & [1] & 1 & 1 & 0 & 0 \\
    0 & 0 & 3 & 3 & 3 & 0 & 0 \\
    0 & 0 & 6 & 6 & 6 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    =
    \begin{bmatrix}
    -1 & -3 & -4 & -3 & -1 \\
    -3 & [-9] & -12 & -9 & -3 \\
    -5 & -15 & -20 & -15 & -5 \\
    3 & 9 & 12 & 9 & 3 \\
    6 & 18 & 24 & 18 & 6
    \end{bmatrix}
    \]

    The origins of the output matrices are determined by the sums computed when the origins of the kernel and image matrices align.

    Edges correspond to high magnitudes in the gradient matrices. $S_x * I_1$ has a column of zeroes, which indicates that the pixel intensity stays the same as $x$ changes. In other words, there are no vertical edges. On the other hand, $S_y * I_1$ only contains non-zero (and larger) magnitudes, so there are horizontal edges in the image. The edge with the greatest magnitude exists at $(m, n) = (3, 1)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(b) Laplacian By Hand

    Compute the full linear convolution \(L * I_2\) **by hand**. Mark the origin in the output matrix. Use the response to locate any corner(s), and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 1(b)

    For a full linear convolution, we need to zero pad $I_2$ with $2$ zeroes in each direction. The Laplacian kernel is identical when flipped, so our computation is actually between $L$ and $I_{2, \text{padded}}$:

    \[
    L * I_{2} = \begin{bmatrix}
    0 & 1 & 0 \\
    1 & [-4] & 1 \\
    0 & 1 & 0
    \end{bmatrix}
    *
    \begin{bmatrix}
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & [10] & 10 & 10 & 0 & 0 \\
    0 & 0 & 10 & 100 & 100 & 0 & 0 \\
    0 & 0 & 10 & 100 & 100 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 & 10 & 10 & 10 & 0 \\
    10 & [-20] & 80 & 70 & 10 \\
    10 & 80 & -180 & -190 & 100 \\
    10 & 70 & -190 & -200 & 100 \\
    0 & 10 & 100 & 100 & 0
    \end{bmatrix}
    \]

    The origin of the output matrix is determined by the sum computed when the origins of the kernel and image matrices align.

    The central region contains large negative responses (−180,−190,−200), which suggests strong local intensity changes. Moreover, we notice that there is a sign change between these negative responses and most of the surrounding pixels, which can indicate a zero crossing. So, there is likely a corner at those pixel locations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(c) Verify in Python

    Complete the two code cells below. The function must perform full 2D linear convolution without calling a library convolution routine (you may not use any numpy functions).
    """)
    return


@app.cell
def _(np):
    def convolve2d_full(image, kernel):
        image = np.array(image, dtype=float)
        kernel = np.array(kernel, dtype=float)

        """Return the full 2D linear convolution of image and kernel."""
        output_shape = (
            image.shape[0] + kernel.shape[0] - 1,
            image.shape[1] + kernel.shape[1] - 1,
        )
        output = np.empty(output_shape, dtype=float)

        # EDIT BELOW: flip kernel and accumulate every shifted product in output.
        # Flip the kernel
        flipped_kernel = kernel[::-1, ::-1]

        # Add zero padding required for full convolution (kernel size - 1 on each side)
        pad_h = (kernel.shape[0] - 1, kernel.shape[0] - 1)
        pad_w = (kernel.shape[1] - 1, kernel.shape[1] - 1)
        padded_image = np.pad(image, (pad_h, pad_w), mode="constant", constant_values=0)

        # Slide the kernel across the padded image and compute the output matrix
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                # Get the matching set of values from the image currently aligned with the kernel
                image_overlap = padded_image[i : i+kernel.shape[0], j : j+kernel.shape[1]]
    
                # Use the flipped kernel and sum all elements in the overlap product matrix
                overlap_matrix = flipped_kernel * image_overlap
                output[i, j] = overlap_matrix.sum()
    
        return output

    return (convolve2d_full,)


@app.cell
def _(convolve2d_full, image_1, image_2, laplacian, sobel_x, sobel_y):
    try:
        print(convolve2d_full(image_1, sobel_x))
        print(convolve2d_full(image_1, sobel_y))
        print(convolve2d_full(image_2, laplacian))
    except Exception as error:
        print(f"Problem 1(c) test is not ready: {error}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Your Convolution on a Real Image

    The next cell downloads a photograph, converts it to grayscale, and runs your convolution function with both Sobel kernels. It displays the input, the horizontal and vertical responses, and their combined edge magnitude. You can use this as a quick check after implementing `convolve2d_full`.
    """)
    return


@app.cell(hide_code=True)
def _(convolve2d_full, mare_image, np, sobel_x, sobel_y):
    try:
        mare_sobel_x = convolve2d_full(mare_image, sobel_x)
        mare_sobel_y = convolve2d_full(mare_image, sobel_y)
        mare_edges = np.hypot(mare_sobel_x, mare_sobel_y)
        mare_error = None
    except Exception as error:
        mare_sobel_x = mare_sobel_y = mare_edges = None
        mare_error = str(error)
    return mare_edges, mare_error, mare_sobel_x, mare_sobel_y


@app.cell(hide_code=True)
def _(mare_edges, mare_error, mare_image, mare_sobel_x, mare_sobel_y, mo):
    try:
        if mare_error:
            mo.output.replace(mo.md(f"Sobel test on mare is not ready: {mare_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack(
                        [mo.image(mare_image, width=220), mo.image(mare_sobel_x, width=220)]
                    ),
                    mo.hstack(
                        [mo.image(mare_sobel_y, width=220), mo.image(mare_edges, width=220)]
                    ),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the Sobel test: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2: Gaussian and Laplacian Pyramids

    You are given a color image. Convert it to grayscale, then crop the 512×512 region beginning at the upper-left corner. Use this crop as the input image for both parts.

    You will build pyramid levels at 512×512, 256×256, 128×128, and 32×32. To reach 32×32 from 128×128, blur and downsample twice. Do not use a prebuilt pyramid implementation.

    A normalized 5×5 Gaussian kernel is provided in the starter code. In your report, state your boundary convention, show the requested pyramid images at their corresponding resolutions, and include your code. If you use any non-standard-library helper, include it with a brief explanation.
    """)
    return


@app.cell
def _(storke_tower_image):
    # EDIT BELOW: take the upper-left crop from storke_tower_image.
    input_image = storke_tower_image
    input_image = input_image[0:512, 0:512] # Crop to 512 x 512
    print(input_image.shape)
    return (input_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(a) Gaussian Pyramid

    Create a Gaussian pyramid from the 512×512 grayscale crop using Gaussian convolution and the provided normalized 5×5 kernel. Apply blur and downsampling to produce four levels, in this order: 512×512, 256×256, 128×128, and 32×32. Document your boundary convention, and include the four images and your code in the report. Do not use a prebuilt pyramid implementation.
    """)
    return


@app.cell
def blur_and_downsample(convolve2d_full):
    def blur_and_downsample(image, kernel):
        """Blur image with kernel, then reduce each spatial dimension by two."""
        # EDIT BELOW: call convolve2d_full, crop to image.shape, then subsample.
        # Document your boundary convention.

        """
        Boundary Convention:
        For an image with size (H, W) and a kernel of size (K_H, K_W), full 2d convolution
        produces an output of size (H + K_H - 1, W + K_W - 1).

        We want to keep the output image the same size as the input image, so we crop back
        to the original (H, W) dimensions. However, to keep the output centered, the boundaries
        are offset by half the kernel dimensions: (K_H - 1) // 2 and (K_W - 1) // 2.
        """

        # Convolve with Gaussian kernel to blur the image
        blurred_img = convolve2d_full(image, kernel)

        # Determine indices for cropping such that the convolution output is centered
        start_r = (kernel.shape[0] - 1) // 2
        start_c = (kernel.shape[1] - 1) // 2

        # Crop the blurred image to the same size as the original image 
        cropped_img = blurred_img[start_r:image.shape[0] + start_r, start_c:image.shape[1] + start_c]
        print(cropped_img.shape[0], cropped_img.shape[1])

        # Downsample by taking every second pixel in both dimensions
        return cropped_img[::2, ::2]

    return (blur_and_downsample,)


@app.cell
def _(blur_and_downsample, np):
    def build_gaussian_pyramid(
        image, level_shapes=((512, 512), (256, 256), (128, 128), (32, 32))
    ):
        """Return Gaussian-pyramid levels with exactly the requested shapes."""
        kernel = np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ) / 256
        print(image.shape)
        pyramid = [image]

        # EDIT BELOW: call blur_and_downsample until each requested level is reached.
        # Keep levels in the same order as level_shapes.
        current = image
        for target_shape in level_shapes[1:]:
            # Repeatedly blur and downsample until the image shape matches the
            # requested level shape
            while current.shape[:2] != target_shape:
                current = blur_and_downsample(current, kernel)
            pyramid.append(current)

        return pyramid

    return (build_gaussian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(b) Laplacian Pyramid

    Using your Gaussian pyramid, construct the corresponding Laplacian pyramid. Construct the Laplacian pyramid for each of the four resolutions given above. For example, to construct the Laplacian pyramid at resolution 512x512, you take the Gaussian smoothed image at resolution 512x512 and subtract it from the original 512x512 image.
    """)
    return


@app.cell
def build_laplacian_pyramid(convolve2d_full, np):
    def build_laplacian_pyramid(gaussian_pyramid):
        """Return Laplacian levels followed by the coarsest Gaussian residual."""
        pyramid = []

        # EDIT BELOW: construct the Laplacian level at each requested resolution,
        # then append the final coarsest Gaussian level.

        # Get the Gaussian kernel again for smoothing
        kernel = np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ) / 256

        for g_img in gaussian_pyramid:
            # Get the Gaussian smoothed image at the current resolution
            blurred_img = convolve2d_full(g_img, kernel)
        
            # Crop back to the original image's exact dimensions
            start_r = (kernel.shape[0] - 1) // 2
            start_c = (kernel.shape[1] - 1) // 2
            cropped_img = blurred_img[start_r : start_r + g_img.shape[0], start_c : start_c + g_img.shape[1]]
        
            # Calculate the Laplacian image and add it to the pyramid
            l_img = g_img - cropped_img
            print(l_img.shape[0], l_img.shape[1])
            pyramid.append(l_img)

        # Append the final coarsest Gaussian level (32x32)
        pyramid.append(gaussian_pyramid[-1])
        
        return pyramid

    return (build_laplacian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pyramid Visualizations

    After completing your functions, the hidden cells below display the Gaussian pyramid in the first row and the Laplacian pyramid in the second row.
    """)
    return


@app.cell
def _(build_gaussian_pyramid, build_laplacian_pyramid, input_image):
    try:
        gaussian_pyramid = build_gaussian_pyramid(input_image)
        laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
        pyramid_error = None
    except Exception as error:
        gaussian_pyramid = laplacian_pyramid = None
        pyramid_error = str(error)
    return gaussian_pyramid, laplacian_pyramid, pyramid_error


@app.cell
def _(gaussian_pyramid, laplacian_pyramid, mo, pyramid_error):
    try:
        if pyramid_error:
            mo.output.replace(mo.md(f"Pyramid visualization is not ready: {pyramid_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack([mo.image(level, width=150) for level in gaussian_pyramid]),
                    mo.hstack([mo.image(level, width=150) for level in laplacian_pyramid]),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the pyramids: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3: Harris Corner Detector

    For this part, implement a Harris corner detector. At each image location, compute the corner response function (CRF)

    \[
    R(x, y) = \det(A) - k\bigl(\operatorname{trace}(A)\bigr)^2,
    \]

    where \(A\) is the local structure matrix

    \[
    A =
    \begin{bmatrix}
    \langle I_x^2 \rangle & \langle I_x I_y \rangle \\
    \langle I_x I_y \rangle & \langle I_y^2 \rangle
    \end{bmatrix}.
    \]

    Use a simple constant 3×3 averaging window for the local averages. You may use any discrete derivative approximation, but write down the equations you use. You may ignore image-border effects by zero-padding or by limiting calculations to the image interior.

    Use the CRF to identify the top \(N\) candidate corners, where \(N\) is around 20–50. Threshold and sort the candidates before selecting the final locations. Your function should return `(row, column)` locations.

    Put your complete implementation, including any helper functions, in the cell below. You may use functions defined elsewhere in this notebook. Use the KAVLI image to demonstrate your detector: include your code in the report, overlay the detected corners on the image, list their coordinates, state your parameter values, and comment on the results.
    """)
    return


@app.cell
def harris_corners(convolve2d_full, np):
    def crop_image(image, kernel, shape):
        # Crops image to the size specified by shape
        start_r = (kernel.shape[0] - 1) // 2
        start_c = (kernel.shape[1] - 1) // 2
        cropped_img = image[start_r : start_r + shape[0], start_c : start_c + shape[1]]

        return cropped_img

    def harris_corners(image, threshold, max_corners=40, k=0.04):
        """Return up to max_corners Harris corners as (row, column) locations."""
        # EDIT BELOW: write your complete detector and any helpers here.

        # Get the image derivatives I_x and I_y. We'll use the simplified equations
        # I_x(r, c) = (I(r, c + 1) - I(r, c - 1)) / 2
        # I_y(r, c) = (I(r + 1, c) - I(r - 1, c)) / 2
        # For the boundary rows and columns, assume zero padding.
        padded = np.pad(image, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)
        i_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
        i_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0

        # Calculate the necessary values based on the gradients for the structure matrix
        i_x_squared = i_x * i_x
        i_crossterm = i_x * i_y
        i_y_squared = i_y * i_y

        # Get the local averages of the structure matrix values using a constant 3x3 averaging window
        # The constant 3x3 averaging kernel used is: [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
        kernel = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=float,
        ) / 9

        # Convolve the i terms with the averaging kernel for the structure matrix A. We also crop the
        # convolved output to the size of the original image
        a_1 = crop_image(convolve2d_full(i_x_squared, kernel), kernel, image.shape)
        a_2 = crop_image(convolve2d_full(i_crossterm, kernel), kernel, image.shape)
        a_4 = crop_image(convolve2d_full(i_y_squared, kernel), kernel, image.shape)

        # Calculate the determinant and trace of A, then calculate the corner response function
        det_a = (a_1 * a_4) - (a_2 * a_2)
        trace_a = a_1 + a_4
        crf = det_a - (k * (trace_a ** 2))

        # The response is large and positive at corners, so only keep the coordinates of 
        # candidates with positive responses. Sort the remaining candidates by descending
        # corner response and store their indices
        candidate_r, candidate_c = np.where(crf > 0)
        candidate_crf = crf[candidate_r, candidate_c]
        sorted_indices = np.argsort(candidate_crf)[::-1]

        # Select the indices of the top N candidates with the highest response. If 
        # N > max_corners, pick the max number of candidates allowed. If N > number 
        # of remaining candidates, select all remaining candidates.
        n = min(int(threshold), max_corners, len(sorted_indices))
        sorted_indices = sorted_indices[:n]

        # Return the coordinates associated with those indices
        corners = [(int(candidate_r[i]), int(candidate_c[i])) for i in sorted_indices]
        return np.asarray(corners)

    return (harris_corners,)


@app.cell
def _(harris_corners, kavli_image, plt):
    # EDIT BELOW: choose a threshold, then plot the detected corners.
    corners = harris_corners(kavli_image, threshold=35)
    print(corners)
    plt.imshow(kavli_image, cmap="gray")
    plt.axis("off")
    plt.scatter(corners[:, 1], corners[:, 0], c="r", s=10)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 4: Philosophy of Computer Vision

    ## 4(a) Biological Facial Recognition

    How do humans recognize faces? Describe the features the visual system could use and the processing that might occur at each stage, from seeing a face to identifying the person.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(a)

    I would break down recognizing faces into two steps: (1) detecting that an object is a human face and (2) recognizing the identity associated with it. First, the brain observes basic structure, observing the edges, corners, and contrast of an object to understand its depth and boundaries. These individual pieces are then combined into familiar facial features based on their shape and positions, piecing together an eye, nose, mouth, etc. The overall configuration of these features relative to one another allows us to recognize a whole face due to our learnt knowledge of what faces look like. Processing identity occurs in a different part of the brain, as it requires noticing details idiosyncratic to each person, rather than detecting generic human features. Broadly, I would guess that we compare a perceived face with all familiar faces in our memory and recognize the person if there is a match.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(b) Computer Vision in the Age of AI

    As AI becomes better at writing code and optimizing pipelines, what should students learn in a computer vision course?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(b)

    I think a computer vision course should focus less on students implementing these pipelines from scratch and more on understanding these systems and evaluating the most efficient solutions. Students should operate more as architects for a problem rather than laborers, making the design decisions on what information needs to be extracted and how features should be detected and classified. It's also still important to have the mathematical and conceptual foundations to explain the algorithms employed by AI, even if they are not writing everything themselves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(c) Course Milestone

    What would you like to accomplish by the end of this course? Describe a project, a skill you want to develop, or another goal you hope to reach.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(c)

    I'm interested in autonomous vehicles and want to learn how to develop perception systems for them. With my existing experience in sensor and peripheral design, I'd like to integrate my hardware knowledge with what I learn from this course to build systems that can detect obstacles in real time and use that information to control a moving platform. I also want to understand the computer vision algorithms commonly used in industry and how engineers select and adapt them to meet practical constraints.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submitting Your Work

    If you used Molab, click **Export** and choose **Python** to download your completed notebook as a `.py` file. Submit that `.py` notebook file to Gradescope.
    """)
    return


if __name__ == "__main__":
    app.run()
