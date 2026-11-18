"""Teaching content for the Learning Lab: glossary, lesson list, and quiz questions."""
from __future__ import annotations

LESSONS = [
    ("l1", "What is a QNN?", 8, "Understand qubits, measurement and what a quantum neural network is."),
    ("l2", "Clean the data first", 10, "Spot and fix messy data, and see why some cleaning must wait until after the split."),
    ("l3", "From numbers to qubits", 8, "See how a number becomes an angle, and an angle becomes a qubit state."),
    ("l4", "The circuit that learns", 10, "Read a circuit and count the angles that training will adjust."),
    ("l5", "Training: watch it learn", 10, "Watch the loss fall, and see how the learning rate changes everything."),
    ("l6", "A fair comparison", 12, "Compare quantum and classical models the honest way."),
    ("l7", "Limits and honest science", 8, "Spot the flaws that make quantum claims unreliable."),
]

# (term, full form, plain-English meaning)
GLOSSARY = [
    ("QNN", "Quantum Neural Network", "A machine-learning model whose “brain” is a quantum circuit with adjustable angles."),
    ("QML", "Quantum Machine Learning", "Using quantum circuits to learn patterns from data."),
    ("Qubit", "Quantum bit", "The basic unit of a quantum computer. A normal bit is 0 or 1; a qubit can be a mix of both until you measure it."),
    ("Superposition", "", "A qubit being a mix of 0 and 1 at the same time."),
    ("Entanglement", "", "A link between qubits so they cannot be described one by one."),
    ("Gate", "Quantum gate", "An operation on qubits, for example a rotation."),
    ("Circuit", "Quantum circuit", "A sequence of gates that ends with a measurement."),
    ("Ansatz", "", "The chosen pattern of adjustable gates (German for “approach”)."),
    ("Variational circuit", "", "A circuit whose gate angles are adjusted by training."),
    ("Encoding", "Data encoding", "Writing numbers into qubit states."),
    ("Measurement", "", "Reading a qubit. You get 0 or 1, with probabilities set by its state."),
    ("Shots", "", "How many times a circuit is run, to estimate probabilities from the results."),
    ("Simulator", "", "Software on a normal computer that imitates a quantum computer."),
    ("NISQ", "Noisy Intermediate-Scale Quantum", "Today's quantum computers: small (tens to hundreds of qubits) and error-prone."),
    ("CSV", "Comma-Separated Values", "A spreadsheet saved as plain text."),
    ("PCA", "Principal Component Analysis", "Mixes many columns into a few summary columns that keep most of the information."),
    ("IQR", "Interquartile Range", "The spread of the middle half of a column. Values far outside it are flagged as extreme."),
    ("Epoch", "", "One full pass over the training data."),
    ("Batch", "Mini-batch", "A small group of rows learned from at once."),
    ("Optimizer", "", "The method that adjusts the model: Adam (Adaptive Moment Estimation), SGD (Stochastic Gradient Descent), RMSprop (Root Mean Square Propagation)."),
    ("Learning rate", "", "The size of each learning step."),
    ("Loss", "", "How wrong the model currently is. Training pushes it down."),
    ("Overfitting", "", "Memorising the training data but failing on new data."),
    ("Data leakage", "", "When information from the test data sneaks into training, making scores look better than they are."),
    ("Baseline", "", "A simple or common model that you compare your new model against."),
    ("MLP", "Multi-Layer Perceptron", "A small classical neural network."),
    ("SVM", "Support Vector Machine", "A classical model that draws the widest possible boundary between classes."),
    ("R²", "R-squared", "A score for predicting numbers. 1 is perfect; 0 is no better than guessing the average."),
    ("Benchmark", "", "A fair test that compares models on the same data."),
]

TAKEAWAYS = [
    "A QNN is a classical learning loop wrapped around a small quantum circuit that runs on a simulator.",
    "Clean first, but fit anything that learns from data (gap filling, scaling, PCA) on the training rows only.",
    "Numbers become rotation angles; more qubits cost more time and do not guarantee better results.",
    "Every adjustable angle is a parameter that training tunes; more layers mean more parameters, not always more accuracy.",
    "Watch the loss: a good learning rate makes it fall smoothly.",
    "Judge a model by a spread over several runs, never by one lucky number.",
    "Competitive is not advantage. Say what you measured, and what you did not.",
]
