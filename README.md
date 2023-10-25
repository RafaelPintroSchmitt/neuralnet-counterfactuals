# Neural network counterfactuals

More information can be found in the slides and paper attached.

In order to run this code, I suggest opening the jupyter notebook on Google Colab in order to use the free GPU. Create a new cell, clone this directory (!git clone "path"), fix the main directory in the file and run. Running this code without GPU's takes about an hour, and 3 minutes with a GPU.

Feel free to contact me for any suggestions or questions.

# Graph Neural Network Counterfactuals

## Overview
In this section, I propose a method to evaluate treatment effects based on a neural network's (lack of) accuracy post-treatment. This comes down to recasting causal inference as a prediction problem. My discussion will be inspired by a conformal inference test. 

## Notation and Framework
- Let $t\in \{1,2,...,T\}$ be a generic time period in a sequence of length $T$.
- Consider $\{Y^I_{i,t}\}_{t=1}^T$ as a sequence of outcomes for a unit of observation $i$ followed over time, under some intervention at time $T_0$.
- Let $\theta_t$ be a scalar capturing the effect of the intervention, with $\theta_t = 0$ for $t < T_0 $.
- $ \{ Y^N_{i,t}\}^T$ denotes $Y^I_{i,t} - \theta_t$ for each $t$, representing a counterfactual world where the intervention did not take place.
- $\{P_{i,t}^N\}$ be a sequence of mean-unbiased predictors or proxies for $Y^N_{i,t}$, such that $Y^N_{i,t} = P_{i,t}^N + u_t$, where $E(u_t)=0$ $\forall$ $t \in $\{1,...,T\}.

Under suitable assumptions, having $P_{i,t}^N$ allows us to test hypotheses on $\theta_t$. One such hypothesis is whether $\theta_t=0$ $\forall$ $t\geq T_0.

## Neural Network Approach
My contribution is to provide a method to create counterfactuals using a graph neural network architecture.

### Back to the Topic
The main issue I encountered when trying to use wind-death timing as a treatment was that cities selected into treatment presented differential trends in prior elections. To tackle this issue, I use a prediction problem approach and a graph attention network (GAT).

### Defining the Neural Network
- I construct a graph for each year, where each city is connected to its 10 closest neighbors.
- The model's objective is to predict the outcome for all cities based on control cities' information.
- The neural network architecture consists of a dropout layer, two attention layers, a ReLU layer, and a linear layer.

## Performance
The model performs well, with $R^2_{GAT}\approx 0.8$ in the training set. It is unbiased from 2010 to 2018 and provides accurate predictions.

## Treatment Effect Evaluation
The neural network's predictions are used as counterfactuals for the treated group. The results indicate a treatment effect in 2022, as the bias in predictions suggests a negative impact of the intervention on worker's party vote share.

### Conclusion
This approach provides a credible counterfactual and supports the argument that excess mortality caused a decrease in Bolsonaro's vote share.

For more details and figures, please refer to the complete document.

