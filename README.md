# AI Labs – Summary
 
## Lab 1 – Genetic Algorithm for Optimization
Implementation of a genetic algorithm to solve a control optimization problem for a two-engine rocket. The algorithm uses roulette wheel selection, single-point crossover, and bit mutation to evolve a population of binary chromosomes toward the optimal objective value. The effect of mutation probability `pm` on performance was studied — very small values (`pm ∈ [10⁻⁵, 10⁻⁴]`) yielded near-optimal results, while high values (`pm ≥ 0.05`) caused the population to disperse and the algorithm to fail to converge.
 
## Lab 2 – Minimax with Alpha-Beta Pruning
Implementation of a Minimax player for the Dots and Boxes game, using alpha-beta pruning to reduce the search space. The evaluation heuristic is the score difference between the two players. Two experiments were conducted: (1) comparing players of different search depths — deeper search did not guarantee better play, because the simple heuristic rewards short-term gains and leads deeper players into traps; (2) checking whether the starting player has an advantage — a slight advantage exists at shallow depths but disappears for deeper search.
 
## Lab 3 – Decision Tree (ID3)
Implementation of the ID3 decision tree algorithm for cardiovascular disease classification. Continuous features were discretized into intervals and one-hot encoded before training. The model was evaluated across 10 random seeds and tree depths from 1 to 20. The optimal depth was found to be 5 (mode across seeds), achieving a test accuracy of 0.7251 ± 0.0038. Training accuracy grew monotonically with depth while validation accuracy peaked and then declined, confirming classic overfitting behavior in unpruned trees.
 
## Lab 4 – Gradient Descent on the Ackley Function
Implementation of gradient descent with a fixed step size, applied to the Ackley function in 1D and 2D. The influence of step size `α` and starting point on convergence was studied. Small values (`α ∈ {0.001, 0.01}`) led to stable convergence to a local minimum, while large values (`α ≥ 0.2`) caused divergence or chaotic oscillations. The global minimum (x=0, f=0) was rarely reached, illustrating that gradient descent offers no guarantees on multimodal functions.
 
## Lab 5 – Multilayer Perceptron (MLP)
From-scratch implementation of an MLP with configurable architecture, sigmoid and ReLU activations, MSE loss, mini-batch SGD with reshuffling, and the Adam optimizer. The model was applied to wine quality classification (11 features, classes 0–10). The best configuration — architecture [32, 16], lr=0.1, batch size 32, 500 epochs — achieved ~0.571 test accuracy. A comparison of SGD and Adam showed that Adam converges faster, is less sensitive to learning rate, and achieves slightly better results out of the box (0.580 vs 0.567). Experiments with extreme hyperparameter settings confirmed the sensitivity of MLPs to underfitting and overfitting.
 
