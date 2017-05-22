---
layout: post
title: Finding Recurrent Weights in Keras Models
comments: true
date: 2017-05-21
---

As a researcher who is trying to gain a better understanding of the internal dynamics of recurrent neural networks, sometimes I want to look at the trained weights of my networks.
To train my networks, I use the library <a href="keras.io" target="_blank">Keras</a>.
In general I am very happy with this library, as it allows me to easily change the architecture of my network, try different layer types and use different optimizers.
Inspecting the weights of Keras networks, however, is not the easiest task, nor is printing the activation of its hidden layers (on which more later).     

Although all layers' weights can be accessed via a function `get_weights()`, for recurrent layers this function returns a  list of arrays storing concatenations of gate, and hidden layer weights (and biases), of which the order is not mentioned.
After a little digging in the source code I found that in the latest release of Keras (which is Keras2 at this moment) the different weight matrices are stored in attributes of the layer.
As I feel that accessing and the weight matrices of networks should be easy (and common practice), I decided to share this with you in my first blog post.  

### Finding weights of a GRU layer

Thus, following the example on the Keras website, we create a network with one GRU layer:

{% highlight python %}
from keras.models import Sequential
from keras.layers import GRU

model = Sequential()
model.add(GRU(10, input_shape=(8, 15), return_sequences=True))
{% endhighlight %}

Now, the last (and only) layer of this network is a GRU layer, whose different weight matrices we can access as follows.

{% highlight python %}
GRU_layer = model.layers[0]
recurrent_weights = GRU_layer.recurrent_kernel_h.eval()
update_gate_weights = GRU_layer.recurrent_kernel_r.eval()
reset_gate_weights = GRU_layer.recurrent_kernel_z.eval()
{% endhighlight %}

The bias and weights from the input to the hidden layer, update- and reset gate are stored in `layer.bias_\{h,r,z\}` and `kernel_\{h,r,z\}`, respectively.

### So, what does the `get_weights()` method return?

Now that we have the values of the different weight matrices, we can find out in which order the weights are concatenated in the output of the  `get_weights()` method.
After running a bunch of statements of the form

{% highlight python %}
np.array_equal(GRU_layer.get_weights()[1][:,10:20], recurrent_weights)
{% endhighlight %}

I conclude that for the `get_weights()` function returns 
{% highlight python %}
GRU.get_weights() = [[W_z; W_r, W_h], [U_z, U_r, U_h], [bias_z, bias_r, bias_h]]
{% endhighlight %}

Well, that was it for now, hope this was useful for anyone (and if not, it will probably be for future me :)).
