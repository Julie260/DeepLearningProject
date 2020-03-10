# DeepLearningProject
#RNN for handwritting generation

Adapted from https://lirnli.wordpress.com/2017/09/24/notes-fake-handwriting-generation-with-pytorch/ and Alex Grave's famous paper https://arxiv.org/abs/1308.0850

## Data

#### IAM database
The IAM online handwriting database (IAM-OnDB) consists of handwritten lines collected from 221 diﬀerent writers using a connected whiteboard. Their pen was tracked and recorded using an infra-red device in the corner of the board.
The original input data contains the x and y coordinates and the points where the pen is lifted. No pre-processing was used except to fill missing readings, and for exceeding lengths. The data used contains the \delta_x and \delta_y between the current and the last points, and an indicator of wether the pen was lifted or not.
The data can be downloaded here [http://www.fki.inf.unibe.ch/databases/iam-handwriting-database].

#### MNIST database
The MNIST database of handwritten digits is mainly made of images of handwritten digits. It also offers files containing tables of coordinates that have been sampled on the images of the digits, enabling to follow mathematically the trail of the writers’ pen.
We use for the training this second part of the dataset, very similar to the IAM database, as it contains the \delta_x and \delta_y from one point to the next one, but also two indicators. One indicates the end of the stroke, the other the end of the character.
Finally, to enable the training for the on-demand generation of handwritten digits, we had to process the data in order to label it thanks to a separated file containing the ordered label of each sample.

The data can be downloaded here [https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/wiki/MNIST-digits-stroke-sequence-data?fbclid=IwAR2H1KkkFyVEam6-cY-oXoujkX2KTw8FIYtcSRBant9BKtIYTAXlGnI8Iug]

## Project Structure

```bash
.
├── loaders
|   └── load_IAM.py #loading and pre-processing IAM data
|   └── load_MNIST.py # loading and pre-processing MNIST data
|   └── load_split_MNIST.py #loading and pre-processing MNIST data for the train split models
|   └── loader.py
├── models
|   └── model.py #create the selected model
|   └── Gaussian_Writing_LSTM.py
|   └── Gaussian_Writing_GRU.py
├── toolbox
|   └── draw_samples.py #to draw the generated samples
|   └── generate_samples.py  # from the output of the net (parameters for a gaussian mixture model and bernoulli), generate the points
|   └── losses.py
├── generate_handwriting.py # from a trained model, generate samples
├── train.py # train a net
├── train_split_MNIST_models
|   └── train_split_MNIST.py #train a model per digit
|   └── generate_digits.py  # generate each digit with the corresponding trained model
├── trained_models #trained models ready to use in generate_handwriting.py
#the files are named this way : LSTM/GRU_IAM/MNIST_nb epoch_ne gaussian_learning rate_optimizer_rnn size_n layers
├── images #some example of generated sequence
```

## Launching

- use train.py to train a model, be careful of changing the paths at the beginning of the file
- use generate_handwriting.py to generate handwriting from a trained model, be careful, you need to change the paths and you need to initialize the corresponding model (GRU or LSTM) with the right parameters (the files are named this way : LSTM/GRU_IAM/MNIST_nb epoch_ne gaussian_learning rate_optimizer_rnn size_n layers)
- use generate_digits.py to generate digits from the trained models in train_split_MNIST_models directory. Be careful to change the paths at the beginning of the file. (here you do not need to change the model parameters)



