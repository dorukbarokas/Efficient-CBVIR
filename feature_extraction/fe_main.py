from torchvision import transforms
from featureextraction.solar.solar_global.networks.imageretrievalnet import extract_vectors
from featureextraction.solar.solar_global.utils.networks import load_network


def extract_features_global(cuda_device,images, net, size=256):
    """Extract features from the given `images`.

    Arguments:
        images:     (numpy)array containing the loaded images
        size:       the size to resize the (shortest size of the) images to

    Returns:
        image_features: tuple containing the extracted features of the
                        `search_images` and the `frame_images`
    """
    # net = load_network('resnet101-solar-best.pth')
    # net.mode = 'test'

    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),  # the FromImageDataList() that is used in extract_vectors uses Tensors
        transforms.Resize(size=size),
        normalize
    ])

    ms = [1, 2**(1/2), 1/2**(1/2)]

    image_features = extract_vectors(cuda_device, net, images, size, transform, ms=ms, mode='test')

    image_features = image_features.transpose(0, 1)

    return image_features
