

Work completed as a research intern in teuscher.:Lab (https://www.teuscher-lab.com/) at Portland State Univeristy (Summer 2025)
## Description
This project implements a fractional-order leaky integrate-and-fire spiking neural network to solve the CartPole-v1 environment. The effects of $\alpha$ variation on the fractional dynamics is studied in relation to the networks efficiency and robustness. The fractional-order network is compared to a first-order spiking neural network and a standard multilayer perceptron. Associated scripts for all model architectures and resulting plots are included.

## Directory:
scripts/ - CartPole MLP, SNN/FOSNN, GL fractional neuron

results/ - Data Tables, Script Plots

## GL Fractional Derivative
Continuous-time Grünwald–Letnikov fractional derivative:

$D^\alpha V(t) = \lim_{h \to 0} \frac{1}{h^\alpha} \sum_{k=0}^{N} (-1)^k \binom{\alpha}{k} V(t - kh)$

Converted to a time-discrete update formula for approximation of membrane voltage:

$V_m(t_{n+1}) = \alpha V_m(t_n) + h^\alpha\cdot\Biggl(\frac{-(V_m(t_n)-V_{rest})+R_mI_{syn}(t_n)}{\tau_m}\Biggl) - \sum_{k=2}^{n+1}\omega_k^{\alpha}V_m(t_{n+1-k})$
    


