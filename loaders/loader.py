from loaders import load_IAM, load_MNIST, load_split_MNIST

def get_loader(name, label=0):
    if name == "IAM":
        return load_IAM.batch_generator()
    if name == "MNIST":
        return load_MNIST.batch_generator()
    if name == "split_MNIST":
        return load_split_MNIST.batch_generator(label)