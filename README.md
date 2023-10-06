
# A mean-field to capture asynchronous irregular dynamics of conductance-based networks of adaptive quadratic integrate-and-fire neuron models

*Accompanying code for the Alexandersen, Christoffer G., et al. "A mean-field to capture asynchronous irregular dynamics of conductance-based networks of adaptive quadratic integrate-and-fire neuron models." bioRxiv (2023): 2023-06.

## Abstract

Mean-field models are a class of models used in computational neuroscience to study the
behaviour of large populations of neurons. These models are based on the idea of representing
the activity of a large number of neurons as the average behaviour of ”mean field” variables.
This abstraction allows the study of large-scale neural dynamics in a computationally efficient and mathematically tractable manner. One of these methods, based on a semi-analytical
approach, has previously been applied to different types of single-neuron models, but never
to models based on a quadratic form. In this work, we adapted this method to quadratic
integrate-and-fire neuron models with adaptation and conductance-based synaptic interactions. We validated the mean-field model by comparing it to the spiking network model. This mean-field model should be useful to model large-scale activity based on quadratic neurons
interacting with conductance-based synaps

## How to cite

Alexandersen, Christoffer G., et al. "A mean-field to capture asynchronous irregular dynamics of conductance-based networks of adaptive quadratic integrate-and-fire neuron models." bioRxiv (2023): 2023-06.

```bibtex
@article{alexandersen2023mean,
  title={A mean-field to capture asynchronous irregular dynamics of conductance-based networks of adaptive quadratic integrate-and-fire neuron models},
  author={Alexandersen, Christoffer G and Duprat, Chloe and Ezzati, Aitakin and Houzelstein, Pierre and Ledoux, Ambre and Liu, Yuhong and Saghir, Sandra and Destexhe, Alain and Tesler, Federico and Depannemaecker, Damien},
  journal={bioRxiv},
  pages={2023--06},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

```

## How to run

### Locally

Fastest, complete control, requires python et al. already set up.

```bash
git clone https://github.com/sandrasaghir/Izhikevich-MF.git
cd Izhikevich-MF
jupyter lab
```

