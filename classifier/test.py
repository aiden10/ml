from network import Network

def load_testing_labels() -> list[int]:
    with open("classifier/data/t10k-labels.idx1-ubyte", "rb") as f:
        # Header
        magic = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")

        labels = list(f.read(num_labels))

    return labels

def load_testing_images() -> list[list[int]]:
    with open("classifier/data/t10k-images.idx3-ubyte", "rb") as f:
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
    testing_images = load_testing_images()
    testing_labels = load_testing_labels()
    
    formatted_images = [[pixel / 255.0 for pixel in image] for image in testing_images]
    
    nn = Network(layers_def=[], load_path="classifier/results/network_record.json")
    correct = 0
    incorrect = 0
    
    for i in range(len(formatted_images)):
        result = nn.forward_pass(formatted_images[i])
        
        prediction = result.index(max(result))
        expected = testing_labels[i]
        
        if prediction == expected: correct += 1
        else: incorrect += 1
        
    print(f"results: {correct}/{incorrect + correct}")
    print(f"accuracy: {round(correct/(correct + incorrect), 2) * 100}%")