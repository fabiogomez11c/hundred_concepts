import numpy as np
import cv2 as cv


# Create a 512 x 512 matrix with random values
img = np.random.randint(0, 256, (512, 512), np.uint8)

# Create an image of opencv
image = cv.merge([img, img, img])

# Show the image
cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()




