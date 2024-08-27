
from skimage import io, transform

# Load the RGB image
image = io.imread('necrosis_muscle.tif')

rescaled_image = transform.rescale(image, scale=0.65, anti_aliasing=True, channel_axis = 2)

io.imsave('rescaled.tif', rescaled_image)


