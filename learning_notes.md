

---

# 🚀 ML & Deep Learning 完整学习笔记

*从字幕提取的深度学习核心知识体系*

---

## 📊 整体统计

- **来源**: YouTube视频字幕（3个视频）
- **总句子数**: 2277
- **覆盖主题**: 8个

---


## Introduction & Overview

**概述**: 本节介绍机器学习和深度学习的基本概念、课程概述以及学习目标。

**核心句子** (114条):

- about machine learning in a way that is accessible to absolute beginners.
- What's up you guys? So welcome to Machine Learning for Everyone. If you are someone who
- is interested in machine learning and you think you are considered as everyone, then this video
- is for you. In this video, we'll talk about supervised and unsupervised learning models,
- concepts as we go. So this here is the UCI machine learning repository. And basically,
- So basically, this command here just reads some CSV file that you pass in CSV has come about comma
- now and talk about this data set. So here I have some data frame, and I have all of these different
- this is actually supervised learning. All right. So before I move on, let me just give you a quick
- little crash course on what I just said. This is machine learning for everyone. Well, the first
- question is, what is machine learning? Well, machine learning is a sub domain of computer science
- and all of them might use machine learning. So there are a few types of machine learning.
- The first one is supervised learning. And in supervised learning, we're using labeled inputs.
- with a certain color. Now in supervised learning, all of these inputs have a label associated with
- to learn about patterns in the data. So here are here are my input data points. Again, they're just
- And finally, we have reinforcement learning. And reinforcement learning. Well, they usually
- there's an agent that is learning in some sort of interactive environment, based on rewards and
- But in this class today, we'll be focusing on supervised learning and unsupervised learning
- and learning different models for each of those. Alright, so let's talk about supervised learning
- when we talk about supervised learning. And this just means we're trying to predict continuous
- supervised learning. Now let's talk about the model itself. How do we make this model learn?
- Or how can we tell whether or not it's even learning? So before we talk about the models,
- let's talk about how can we actually like evaluate these models? Or how can we tell
- bar to kind of talk about some of the other concepts in machine learning. So over here,
- that's the whole point of supervised learning is we can compare what our model is outputting to,
- what's the difference here, in some numerical quantity, of course. And then we make adjustments,


## Machine Learning Basics

**概述**: 机器学习是人工智能的一个分支，让计算机从数据中学习模式，而不需要明确的编程规则。

**核心句子** (392条):

- they just have a ton of data sets that we can access. And I found this really cool one called
- the magic gamma telescope data set. So in this data set, if you want to read all this information,
- So in order to do this, we're going to come up here, go to the data folder. And you're going
- to click this magic zero for data, and we're going to download that. Now over here, I have a colab
- I'm just going to call this the magic data set. So actually, I'm going to call this for code camp
- to order I'm just going to, you know, let you guys know, okay, this is where I found the data set.
- So I've copied and pasted this actually, but this is just where I found the data set.
- Okay, great. Now in order to label those as these columns down here in our data frame.
- separated values, and turns that into a pandas data frame object. So now if I pass in a names here,
- then it basically assigns these labels to the columns of this data set. So I'm going to set
- this data frame equal to DF. And then if we call the head is just like, give me the first five things,
- we have G and H. So if I actually go down here, and I do data frame class unique,
- it's one item in our data set, it's one data point, all of these things are kind of the same
- and features are just things that we're going to pass into our model in order to help us predict
- 10 different features. So I have 10 different values that I can pass into some model.
- that focuses on certain algorithms, which might help a computer learn from data, without a
- programming. So you might have heard of AI and ML and data science, what is the difference between
- Now machine learning is a subset of AI that tries to solve one specific problem and make predictions
- using certain data. And data science is a field that attempts to find patterns and draw insights
- from data. And that might mean we're using machine learning. So all of these fields kind of overlap,
- models and to learn outputs of different new inputs that we might feed our model. So for example,
- down here have something in common, that's finding some sort of structure in our unlabeled data.
- first. So this is kind of what a machine learning model looks like you have a bunch of inputs
- that are going into some model. And then the model is spitting out an output, which is our prediction.
- categorical data, there's either a finite number of categories or groups. So one example of a


## Neural Networks

**概述**: 神经网络受人脑启发，由互相连接的神经元组成，通过权重传递信号。

**核心句子** (148条):

- like this. So you have an input layer, this is where all your features would go. And they have
- all these arrows pointing to some sort of hidden layer. And then all these arrows point to some
- sort of output layer. So what is what is all this mean? Each of these layers in here, this is
- something known as a neuron. Okay, so that's a neuron. In a neural net. These are all of our
- the pregnancy, the BMI, the age, etc. Now all of these get weighted by some value. So they
- feature. So these two get multiplied. And the sum of all of these goes into that neuron. Okay,
- input into the neuron. Now I'm also adding this bias term, which just means okay, I might want
- I don't know. But we're going to add this bias term. And the output of all these things. So
- the sum of this, this, this and this, go into something known as an activation function,
- okay. And then after applying this activation function, we get an output. And this is what a
- neuron would look like. Now a whole network of them would look something like this.
- So I kind of gloss over this activation function. What exactly is that? This is how a neural net
- the some sort of weight times these input layer a bunch of times. And then if we were to go back
- we wouldn't. So the activation function is introduced, right? So without an activation
- greater than zero is linear. So with these activation functions, every single output of a neuron
- that the input into the next neuron is, you know, it doesn't it doesn't collapse on itself, it doesn't
- different slopes with respect to some value. Okay, so the loss with respect to some weight
- this step becomes. Now stick with me here. So my new value, this is what we call a weight update,
- times whatever this arrow is. So that's basically saying, okay, take our old w zero, our old weight,
- diverge. But with all of these weights, so here I have w zero, w one, and then w n. We make the same
- weight. So that's how back propagation works. And that is everything that's going on here. After we
- all the all the weights to something adjusted slightly. And then we're going to calculate the
- just, you know, what we've seen here, it just goes one layer to the next. And a dense layer means that
- a dense layer means that all of them are interconnected. So here, this is interconnected with all of these
- So we're going to create 16 dense nodes with relu activation functions. And then we're going


## Deep Learning

**概述**: 深度学习是使用多层神经网络的机器学习方法，能够自动学习数据的层次化特征。

**核心句子** (29条):

- I wanted to talk about is known as a neural net or neural network. And neural nets look something
- features that we're inputting into the neural net. So that might be x zero x one all the way through
- and factor that all out, then this entire neural net is just a linear combination of these input
- literally just write that out in a formula, why would we need to set up this entire neural network,
- how long it takes for our neural net to converge. Or sometimes if you set it too high, it might even
- libraries that we use, right, we've already seen SK learn. But when we start going into neural
- this line here is basically saying, Okay, let's create a sequential neural net. So sequential is
- neural net output. And you'll see that okay, the the F ones are the accuracy gives us 87%. So it
- here, instead of this temperature regressor, I'm going to use the neural net regressor.
- propagation to train a neural net node, whereas in the other one, they probably are not doing that.
- so now what would happen if we use a neural net, a real neural net instead of just, you know,
- it's okay. So my point is, though, that with a neural net, I mean, this is not brilliant, but also
- that with a neural net anymore. But one thing that we can measure is hey, what is the mean squared
- and the neural net. Okay, so this is my linear and this is my neural net. So if I do my neural net
- And this is my mean squared error for the neural net. So that's interesting. I will debug this live,
- a tuple containing one element, which is a six. Okay, so it's actually interesting that my neural
- Let's add a legend. So you can see that our neural net for the larger values, it seems like
- But yeah, so we've basically used a linear regressor and a neural net. Honestly, there are
- sometimes where a neural net is more appropriate and a linear regressor is more appropriate.
- better than a neural net. But for example, with the one dimensional case, a linear regressor would
- and importance of machine learning and neural networks to the present and to the future.
- But what I want to do here is show you what a neural network actually is,
- What we're going to do is put together a neural
- There are many many variants of neural networks,
- As the name suggests neural networks are inspired by the brain, but let's break that down.


## Training Process

**概述**: 训练过程包括前向传播、计算损失、反向传播和参数更新等步骤。

**核心句子** (127条):

- and we can all as a community learn from this together. So with that, let's just dive right in.
- So this means whatever input we get, we have a corresponding output label, in order to train
- penalties. So let's think of a dog, we can train our dog, but there's not necessarily, you know,
- and that's what we call training. Okay. So then, once you know, we've made a bunch of adjustments,
- single time after we train one iteration, we might stick the validation set in and see, hey, what's
- the loss there. And then after our training is over, we can assess the validation set and ask,
- hey, what's the loss there. But one key difference here is that we don't have that training step,
- here is pretty far from you know, this truth that we want. And so this loss is going to be high. In
- to this one. So that might have a loss of 0.5. And then this one here is maybe further than this,
- any point during the training process. Okay. And that loss, that's the final reported performance
- So this would give a slightly higher loss than this. And this would even give a higher loss,
- of describing things. So here are some examples of loss functions and how we can actually come
- up with numbers. This here is known as L one loss. And basically, L one loss just takes the
- value is a function that looks something like this. So the further off you are, the greater your losses,
- then your loss for that point would be 10. And then this sum here just means, hey,
- everything is. Now, we also have something called L two loss. So this loss function is quadratic,
- the the difference between the two. Now, there's also something called binary cross entropy loss.
- loss that we use. So this loss, you know, I'm not going to really go through it too much.
- But you just need to know that loss decreases as the performance gets better. So there are some
- Okay, so the next thing that we're going to do here is we are going to create our train,
- the standard scalar from sk learn. So if I come up here, I can go to sk learn dot pre processing.
- that will help us do that. It's so I'm going to go to from in the learn dot oversampling. And I'm
- is train and then x train, y train. Oops, what's going on? These should be columns. So basically,
- what I'm doing now is I'm just saying, okay, what is the length of y train? Okay, now it's
- what is going on? Oh, it's because we already have this train. So I have to go come up here and split


## Algorithms & Techniques

**概述**: 主要包括监督学习（分类、回归）、无监督学习（聚类）和各种优化算法。

**核心句子** (58条):

- that is something known as classification. Now, all of these up here, these are known as our features,
- Now there's also unsupervised learning. And in unsupervised learning, we use unlabeled data
- there are some different tasks, there's one classification, and basically classification,
- Hot dog, pizza, ice cream. This is something known as multi class classification. But there's also
- binary classification. And binary classification, you might have hot dog, or not hot dog. So there's
- isn't binary classification. Okay, so yeah, other examples. So if something has positive or negative
- sentiment, that's binary classification. Maybe you're predicting your pictures of their cats or
- dogs. That's binary classification. Maybe, you know, you are writing an email filter, and you're
- trying to figure out if an email spam or not spam. So that's also binary classification.
- Now for multi class classification, you might have, you know, cat, dog, lizard, dolphin, shark,
- maybe different plant species. But multi class classification just means more than two. Okay,
- and binary means we're predicting between two things. There's also something called regression
- It looks something like this. And this is for binary classification, this this might be the
- classification because all of our points all of our samples have labels. So this is a sample with
- say, hey, print out this classification report for me. And let's check, you know, I'm giving you the
- that we can expand Bayes rule and apply it to classification. And this is what we call naive
- classification. So that's where this comes in our y hat, our predicted y is going to be equal to
- of misclassification. Right. So that is MAP. That is naive Bayes. Back to the notebook. So
- 72%. Okay. Which, you know, is not not that great. Okay, so let's move on to logistic regression.
- y. Okay. So many of you guys are familiar with regression. So let's start there. If I were to
- draw a regression line through this, it might look something like like this. Right? Well, this
- are doing classification, not regression. Okay. Well, first of all, let's start here, we know that
- which is the y intercept, right? And m is the slope. But when we use a linear regression,
- is it actually y hat? No, it's not right. So when we're working with linear regression,
- one feature x, and that's what we call simple logistic regression. But then if we have, you know,


## Practical Implementation

**概述**: 使用TensorFlow、PyTorch等框架实现机器学习模型。

**核心句子** (170条):

- Kylie Ying has worked at many interesting places such as MIT, CERN, and Free Code Camp.
- Without wasting any time, let's just dive straight into the code and I will be teaching you guys
- actually records certain patterns of you know, how this light hits the camera. And we can use
- properties of those patterns in order to predict what type of particle caused that radiation. So
- know, some length, width, size, asymmetry, etc. Now we're going to use all these properties to
- magic example. Okay. So with that, I'm going to first start with some imports. So I will import,
- you know, I always import NumPy, I always import pandas. And I always import matplotlib.
- And then we'll import other things as we go. So yeah,
- And in order to import that downloaded file that we we got from the computer, we're going to go
- let's give it a number. And this makes sense. Because, like, for example, the thing that I
- Or it might be what is the price of this house? Right? So these things don't really fit into
- So here, all of these are quantitative features, right, because they're all on some scale.
- And as I mentioned, this is what we would call a feature vector, because these are all of our
- because it's even more off. In computer science, we like formulas, right? We like formulaic ways
- So now this is all numerical, which is good, because our computer can now understand that.
- right? It's called so let's just use that might be less confusing of everything up to the last
- and then 50 of another type, well, if you drew the histograms, it would be hard to compare because
- on here and make that the label, the y label. So because it's density, the y label is probability.
- and then get those values. Now, in, so I'm actually going to import something known as
- And I'm going to import standard scalar, I have to run that cell, I'm going to come back down here.
- And now I'm going to create a scalar and use that skip or so standard scalar.
- same as literally doing this. But the negative one is easier because we're making the computer
- so that these kind of match better. And surprise, surprise, there is something that we can import
- going to import this random oversampler, run that cell, and come back down here. So I will actually
- switch oversample here to false. Now, the reason why I'm switching that to false is because my


## Applications

**概述**: 应用包括图像识别、语音识别、自然语言处理、推荐系统等。

**核心句子** (48条):

- values for each entry. Now this is a you know, each of these is one sample, it's one example,
- thing when I mentioned, oh, this is one example, or this is one sample or whatever. Now, each of
- up here, and then it has the class. Now what we're going to do in this specific example is try to
- them, this is the output that we might want the computer to be able to predict. So for example,
- images, they're just pixels. Well, okay, let's say I have a bunch of these different pictures.
- the example, I know this might be a little bit outdated. Here we have a girl and a boy, there are
- example might be okay, we have, you know, a bunch of different nationalities, maybe a nationality or
- So for example, if your input were from the US, you would you might have 1000. India, you know,
- Easter eggs I collected in my basket, this Easter egg hunt, that is an example of discrete quantitative
- with a number that you know, is on some sort of scale. So some examples. So some examples might
- other measures of accurate or performance as well. So for example, accuracy, what is accuracy?
- that into a chart, for example, and this might be my positive and negative tests, and this might
- percentage? What is the chance that this image is a cat? How many cats do I have? Right. And then this
- are all independent. So in my soccer example, you know, the probability that we're playing soccer,
- Right? It's, it's iffy. Okay. For example, we might say, okay, well, it seems like, you know,
- classes. Let's see a few examples. Okay, so first, between these three lines, let's say A, B, and C,
- so the issue with SVM sometimes is that they're not so robust to outliers. Right? So for example,
- rate as x increases, then you're probably looking at something linear. So what's the example of a
- same as this spread over here. Now, what's an example of where you know, homoscedasticity is
- to make this example simpler, I'm just going to index on an hour, and I'm gonna say, okay,
- So in this example below, I have a bunch of scattered points. And you'll see that this
- for example, then okay, this seems like it could be a cluster. This seems like it could be a
- What do I mean by that? So let's take this point here, for example, so I'm computing
- some stable point where nothing is changing anymore. Alright, so that's our first example
- Okay. So for example, if this were something to do with housing prices, right,

