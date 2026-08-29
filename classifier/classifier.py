from network import Network

def load_labels() -> list[int]:
    with open("classifier/data/train-labels.idx1-ubyte", "rb") as f:
        # Header
        magic = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")

        labels = list(f.read(num_labels))

    return labels

def load_images() -> list[list[int]]:
    with open("classifier/data/train-images.idx3-ubyte", "rb") as f:
        # Header
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_columns = int.from_bytes(f.read(4), "big")

        pixels_per_image = num_rows * num_columns

        images = []

        for _ in range(num_images):
            image = list(f.read(pixels_per_image))
            images.append(image)

    return images

if __name__ == "__main__":
    training_images = load_images()
    training_labels = load_labels()

    # Input is list of size n where n is amount of pixels per image
    # Output is 10 because there are 10 possible digits an image can be
    # Not sure how many hidden layers there should be so I'll just try a layer of 10 for now
    layers_definition = [len(training_images[0]), 10, 10]

    formatted_labels = []
    for label in training_labels:
        label_array = [0.0 for _ in range(10)]
        label_array[label] = 1.0
        formatted_labels.append(label_array)
    
    formatted_images = [[pixel / 255.0 for pixel in image] for image in training_images]
        
    nn = Network(layers_definition)
    nn.train(inputs=formatted_images, expected=formatted_labels)