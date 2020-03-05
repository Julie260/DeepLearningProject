from loaders import load_IAM

def get_loader(name):
    if name == "IAM":
        return load_IAM.batch_generator()