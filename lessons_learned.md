# Lessons Learned


In this document, I will share some of the most pertinent issues I learned about Deep Learning, Keras, and Tensorflow while doing this project. Hopefully this helps you avoid the same mistakes, as well as myself to not repeat these again in the future :)


## Deep Learning Concepts



## Keras 


1. Binary Classification
    Whenever you're doing a binary classification problem (such as in this project, where the task is to label an image as "fire" or "no fire", it's crucial to have the **following 3 elements in your model**:
        - the final output layer should be an MLP with 1 output neuron
        - the activation function of this layer should return a value between 0 and 1. As a result, it is conventional to use the ```"sigmoid"``` function here.
        - the loss function of the model should be ```keras.losses.binary_crossentropy```


## Tensorflow


1. Solving How to solve the ```InvalidArgumentError```:

    **Error Message**: ```InvalidArgumentError:  BiasGrad requires tensor size <= int32 max```
    
    **Explanation**: As I understand it, Tensorflow constricts the number of elements that can be in a single tensor to what can be stored in a 32-bit integer. For context, the ```int32``` data type (which is how a 32-bit integer is represented in Tensorflow), can store about 2 billion different values and some change. 
    
    **Resolution**: [this Stack Overflow answer](https://stackoverflow.com/questions/60414562/how-to-solve-the-biasgrad-requires-tensor-size-int32-max-invalidargumenterr) gives an amazing explanation as to the different factors that go into the size of a tensor. For myself, the resolution came by decreasing the number of nodes I was placing in each of the neural network layers.     


## Miscellaneous 