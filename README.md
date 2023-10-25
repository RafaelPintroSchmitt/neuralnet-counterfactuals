# Neural network counterfactuals

More information can be found in the slides and paper attached.

In order to run this code, I suggest opening the jupyter notebook on Google Colab in order to use the free GPU. Create a new cell, clone this directory (!git clone "path"), fix the main directory in the file and run. Running this code without GPU's takes about an hour, and 3 minutes with a GPU.

Feel free to contact me for any suggestions or questions.

# Graph Neural Network Counterfactuals

## Table of Contents
- [Introduction](#introduction)
- [Potential Outcomes Framework](#potential-outcomes-framework)
- [Back to the Topic](#back-to-the-topic)
- [Defining the Neural Network](#defining-the-neural-network)
- [Performance](#performance)
- [Treatment Effect Evaluation](#treatment-effect-evaluation)

## Introduction
In this section, I propose a method to evaluate treatment effects based on a neural network's (lack of) accuracy post-treatment. This comes down to recasting causal inference as a prediction problem. I refer to [Chernozhukov et al (2021)](https://doi.org/10.1080/01621459.2021.1920957), for a summary of the related literature and a general framework that includes the method presented here. My discussion will be inspired by their conformal inference test.

### Potential Outcomes Framework
Following the potential outcomes framework of [Neyman (1923)](#) and [Rubin (1974)](#), and borrowing the notation from [Chernozhukov et al (2021)](https://doi.org/10.1080/01621459.2021.1920957), let $t\in {1,2,...,T}$ be a generic time period in a sequence of length $T$. I consider $Y^I_{i,t}$ to be a the outcome for a unit of observation $i$ followed over time $t$, under some intervention at time $T_0$. Let $\theta_t$ be a scalar capturing the effect of the intervention, with $\theta_t = 0$ for $t < T_0$. $Y^N_{i,t}$ denotes $Y^I_{i,t} - \theta_t$ for each $t$, that is, it represents a counterfactual world where the intervention did not take place. 

Finally, let $\{P_{i,t}^N\}$ be a sequence of mean-unbiased predictors or proxies for $Y^N_{i,t}$, such that $Y^N_{i,t} = P_{i,t}^N + u_t$, where $E(u_t)=0$ $\forall$ $t \in \{1,...,T\}$. The potential outcomes can thus be written as:

$$
Y^N_{i,t} = P_{i,t}^N + u_t
$$

$$
Y^I_{i,t} = P_{i,t}^N + \theta_t + u_t
$$

Under suitable assumptions, having $P_{i,t}^N$ allows us to test hypotheses on $\theta_t$. One such hypothesis, which we want to reject, is whether $\theta_t=0$ $\forall$ $t\geq T_0$. That is, whether we have evidence that the intervention had some effect.

There are many ways to try and obtain $P_{i,t}^N$. One such method is using synthetic controls, in which the counterfactual of some unit is constructed by taking a linear combination of other units, with the goal of mimicking the behavior of the dependent variable of interest as well as possible. If such a method is able to extrapolate from the training sample, i.e. to predict accurately the dependent variable for $t$ outside the time periods used for training, then it provides a good candidate for $P_{i,t}^N$ . Indeed, we can fit the model before an intervention, and use the predictions of the model as counterfactuals after it. My contribution will be to provide a method to create such counterfactuals (i.e. to get a plausible sequence $P_{i,t}^N$, using a graph neural network architecture.

## Back to the Topic
You can find more information on the attached paper, but the main issue I encountered when trying to use (non-net) wind-death timing as a treatment/instrument, was that cities selected into treatment - i.e. $\text{WDT}_{2021}$ - presented differential trends in prior elections. Let us consider, for the purposes of this section, that cities with WDT above the median in their state are "treated," and assign them to "control" otherwise. How can we get an appropriate counterfactual if the treated and control groups are substantially different and their voting patterns do not evolve in parallel?

The two groups still carry information about each other. Traditionally, one would perform some kind of matching (e.g. using propensity scores), so that treated and control units are paired according to some underlying measure of similarity. Though at its core my approach does match treated and (a function of the) control units, I try to tackle the issue as a prediction problem: using only the control units, I attempt to construct a model that approximates the treated units' outcomes as well as possible.

To do so, I use a graph attention network (GAT). Explaining GATs from scratch is outside the scope of this project, but I will try to provide some intuition as I go. The technicalities of the machine learning algorithm are not necessary to understand the application presented here.

### Defining the Neural Network
The first step is setting up the data structure which will be fed into the model. I construct a graph - also called a network - for each year. Each city is connected to its 10 closest neighbors in terms of geographical distance, which is itself used as an edge attribute (a weight used by the neural network). I include the covariates listed in the paper as node attributes, together with the year-percentile of the difference in PT vote-share relative to the previous election, which is the outcome of interest.

The objective of the model is simple: given a sample of Brazilian cities' changes in PT vote share, it should predict the outcome for all other cities. If it learns to do so, we can hope that by inputting the control cities' information, the model will be able to predict the treated cities' outcomes. Note that the model learns a generic instruction: given any set of cities, predict the rest. But by learning to do so, it becomes well-suited for our objective, which is creating a counterfactual for the treated group. I split the sample into training and test sets. The model's parameters are obtained from one set of cities, and my results are derived by applying the model to a different set.

Now I can explain the neural network's architecture - i.e. how it learns. I start with an overview for the reader unfamiliar with neural networks and then go into the specifics. At the first epoch (or training round), I feed the model with the 2010 network. Each layer of the network is basically a set of instructions to receive, transform and transmit the vector/matrix received from the previous layer. Then, the model applies an optimizer step, more precisely a variant of gradient descent called Adam. 

Summarizing:

- Architecture:
    1. A dropout layer (with varying dropout rates).
    2. Two attention layers, with a ReLU in between.
    3. A linear layer.

- Training:
    1. Perform a forward pass and take an optimizer step for each year in the training data (2010-2018).
    2. 100 training rounds for each year - 300 forward passes and optimization steps.

### Performance
The model performs reasonably well. Be reminded that the outcome of interest is the percentile of the difference in worker's party vote share for each year. Also, let us define $R^2_{GAT} = 1 - \frac{MSE_M}{MSE_R}$, where $MSE_M$ is the mean-squared error of the model - in terms of percentile predicted - and $MSE_R$ is the mean-squared error under random guesses. 

I trained the model, 300 forward passes and optimization steps, a hundred times, obtaining different parametrizations for each. The testing is simple: I take the set of cities in the test set, which were not used in training, select half at random and make their outcome variable equal to zero. Then, I input the resulting dataset into the model and evaluate it on its ability to guess the dropped cities' outcomes correctly.

On average, $R^2_{GAT}\approx 0.8$ in the training set, and its accuracy - getting the percentile exactly right - is 2 to 3 times higher than random guesses. The performance on training and test sets is similar, which indicates overfitting is likely not an issue, and the model is on average unbiased from 2010 to 2018. I will come back to this last point.

### Treatment Effect Evaluation
We can finally turn to the evaluation of treatment effects using the neural network's prediction as counterfactuals. Be reminded that I define as "treated" those cities with wind-death covariance above the median in their state.

For the years 2010 through 2022, I omit the outcomes of the treated cities and feed the data into the network. I get predictions for each treated city which are based only on the control cities (and covariates of the treated cities). For each year, I record the bias of the predictions (average error across cities) and repeat the training and evaluation 100 times. I perform the same exercise randomly selecting "treatment" units at every training procedure and year, as a placebo test.

The results are plotted in the following picture:

![alt text]([https://github.com/RafaelPintroSchmitt/neuralnet-counterfactuals/blob/main/outputs/2010_2018.jpg])

The model is roughly unbiased before treatment for the treated, and unbiased before and after treatment for the placebo. That is, it creates fairly good counterfactuals for the treated group before treatment. Using the notation introduced at the beginning of the section, the neural networks' predictions are a good candidate for a sequence $\{P_{i,t}^N\}$ of mean-unbiased predictors or proxies for $Y^N_{i,t}$.

The bias is positive for 2022. This means that the predictions of the model - our counterfactual - overshot the worker's party vote share in treated cities. From the potential outcomes equations, we get that $\theta_{2022}$, the treatment effect, should be negative. Knowing that WDT negatively impacted excess mortality in 2021, we can revert back to the IV reasoning of previous sections to argue that excess mortality caused a decrease in Bolsonaro's vote share. Note, however, that I tackled the issue of selection of cities into treatment, particularly when using WDT instead of NWDT, by constructing a credible counterfactual.

From the error bars for the placebo, we can see that, across the training procedures, a random selection of cities into treatment would generate results as extreme as the ones observed for the treatment group less than 5% of the time.

Furthermore, in the spirit of the conformal test proposed by [Chernozhukov et al (2021)](https://doi.org/10.1080/01621459.2021.1920957), I provide some back-of-the-envelope calculations on how likely it would be to find results as extreme as the ones presented here if there were no effect to be found. Let \{0,0,0,1\} represent the current result, indicating that I do not find effects for the first three years and do find an effect for the fourth. This configuration has a 25% chance of occurring under the set of all permutations of the observed results, which can be interpreted as a sort of p-value.

Taking the placebo in conjunction with the permutations test, it is unlikely that the results here do not reflect a treated-group-specific effect in 2022. However, the same caveat from before applies: it could be that the treated group is different from the control across some dimension that moderates the pandemic's impact on the elections, and the neural network did not have the opportunity to learn this pattern.

One may argue that there should be a validation period of the model before treatment. That is, that training should be conducted only in 2010 and 2014, so we can test whether differentials appear already in 2018 or only in 2022. However, note that the training was conducted in one subset of cities for each year, and the figure refers to applying the model to another subset. No city-year was used in both training and results. Also, there does not seem to be an issue for the model to perform well in 2022 when considering placebo treatments. Nonetheless, I present the results derived from training the network only on data from 2010 and 2014. The main patterns do not change.

![alt text]([https://github.com/RafaelPintroSchmitt/neuralnet-counterfactuals/blob/main/outputs/2010_2014.jpg])