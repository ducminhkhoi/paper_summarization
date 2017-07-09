# [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979v1)

## Summary
This paper presents a summary of methods about interpreting and understanding Deep Neural Networks (DNN) so far. This approach is different from other methods of visualizing layer by layer of DNN. This approach just considers the inner layer of DNN as a blackbox, we just interpreting the input as well as the final layer by adding another regularizer.

The paper first discusses some terminologies like interpretation and explanation.
+ Interpretation: mapping an abstract concept (word embedding, output of before softmax layer in CNN) into domain that human can make sense of (e.g input domain: images, texts,...)
+ Explanation: collection of features of interpretable domain (e.g: heatmap in image or highlighted text) that have contributed for a given example to produce a decision (classification or regression). Explanation can be coarse-grained or fine-grained

### Approaches:
#### Interpreting DNN Model
Build a prototype in input domain that is interpretable and representative of abstract concept by using maximization activation framework (AM)

\[\max_{x}{\log p(\omega_c | x) - \lambda \|x\|^2}\]

$l_2$ norm regularizer can be replaced by data density model (an expert) $p(x)$, can be Gaussian RBM.

\[\max_{x}{\log p(\omega_c | x) - \log {p(x)}}\]

Or can be performed in code space $z$, using generator $g$ in GAN

\[\max_{z \in Z}{\log p(\omega_c | g(z)) -\lambda \|z\|^2}\]

#### Explaining DNN decision
View a data point $x$ as a collection of features $x_i$, assign each of these features a score $R_i$ determining how relevant the feature $x_i$ is explaining function value (final decision) $f(x)$

+ Sensitivity Analysis: The most relevant input features are those to which output is most sensitive. It answers the question: ``What makes this image more/less a car`` rather than ``what makes this image a car``
   \[R_i(x) = (\frac{\partial f}{\partial x_i})^2\]

+ Simple Taylor Decomposition, especially with ReLU activation:
   \[f(x) = \sum_{i = 1}^{d}{R_i(x)} \Leftrightarrow R_i(x) = \frac{\partial f}{\partial x_i} \cdot x_i\]

+ Relevance Propagation: algorithm starts at the output of the network and move in the graph in reverse direction, redistribute the predicting score (total relevance) until input is reached. (Relevance Conservation principle)
  \[\sum_{i = 1}^{d}{R_i = \ldots = \sum_{j}R_j = \sum_{k}R_k = \ldots = f(x)}\]

  They propose a technique called Layer wise Relevance Propagation (**LRP**). They have $\alpha\beta$-rule. VGG: LRP-$\alpha_2\beta_1$, Inception-V3 $\alpha_1\beta_0$
  \[a_k = \sigma(\sum_j{a_jw_{jk}} + b_k)\]
  \[R_j = \sum_k{\frac{a_jw_{jk}^+}{\sum_ja_jw_{jk}^+}}R_k^{\wedge} + \sum_k{\frac{a_jw_{jk}^-}{\sum_ja_jw_{jk}^-}}R_k^{\vee}\]

They also discuss about some trick and recommendation for LRP like apply LRP to successful model in the past, use ReLU activation, and not too many FC (should use dropout), prefer sum (average) pooling to max pooling, and force bias $\leq 0$ at training time. Try $\alpha=1, \beta = 0$ first, if negative relevance is needed or heatmap is too diffuse, use $\alpha=2, \beta=1$, and then use pixel flipping. Then, they give example of implementing LRP-$\alpha_1\beta_0$
\[R_j = \sum_k{\frac{a_jw_{jk}^+}{\sum_ja_jw_{jk}^+}}R_k\]
by using ``forward`` and ``backward`` operation in training DNN.

### Evaluation
Finally, they evaluate their proposed method in 2 aspects: Explanatory Continuity and Explanation Selectivity, the baselines are Sensitive Analysis and Simple Taylor Decomposition to show their advantage.

## Comments
This paper is a good one. Give the comprehensive way to  understanding of DNN. But it also lacks some important techniques listed bellow.

Other approach of interpreting the DNN that the paper does not mention is visualizing layer by layer. That can be done by visualizing the output of each convolution layer or visualizing the kernels (filters) after training process completed. These methods are discussed in the website: http://cs231n.github.io/understanding-cnn/

So, to summary about the methods of visualizing and understanding DNN. We have several approaches:
+ Blackbox approach:
  - Interpreting DNN Model (mentioned)
  - Retrieving images that maximally activate a neuron
  - Embedding the code with TSE
  - Occluding parts of the image (similar to pixel-flipping)
+ Non-blackbox approach
  - Explaining DNN decision (mentioned)
  - Layer activation
  - Conv/FC filters
  - Deconvolution Net
  - Guided Backprogation (or excitation Backprogation)
