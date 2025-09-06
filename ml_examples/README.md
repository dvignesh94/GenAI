# Machine Learning Examples

This repository contains comprehensive examples of the three main types of machine learning: **Supervised Learning**, **Unsupervised Learning**, and **Reinforcement Learning**. Each example is designed to be educational and demonstrates key concepts with real datasets and visualizations.

## 📁 Repository Structure

```
ml_examples/
├── supervised_learning_example.py      # Classification and Regression examples
├── unsupervised_learning_example.py    # Clustering and Dimensionality Reduction
├── reinforcement_learning_example.py   # Q-Learning and Policy Gradient
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Examples

```bash
# Supervised Learning
python supervised_learning_example.py

# Unsupervised Learning
python unsupervised_learning_example.py

# Reinforcement Learning
python reinforcement_learning_example.py
```

## 📊 Examples Overview

### 1. Supervised Learning (`supervised_learning_example.py`)

**What it demonstrates:**
- **Classification**: Using Random Forest to classify Iris flowers
- **Regression**: Predicting house prices using the Boston housing dataset
- Model evaluation metrics and feature importance analysis

**Key Concepts:**
- Train/test split
- Model training and evaluation
- Feature importance
- Performance visualization

**Outputs:**
- `supervised_classification_results.png` - Classification results and feature importance
- `supervised_regression_results.png` - Regression results and residual analysis

### 2. Unsupervised Learning (`unsupervised_learning_example.py`)

**What it demonstrates:**
- **Clustering**: K-Means and Hierarchical clustering on Iris dataset
- **Dimensionality Reduction**: PCA and t-SNE for data visualization
- **Elbow Method**: Finding optimal number of clusters

**Key Concepts:**
- Clustering algorithms comparison
- Dimensionality reduction techniques
- Cluster evaluation metrics (Silhouette Score, ARI)
- Data visualization in reduced dimensions

**Outputs:**
- `unsupervised_clustering_results.png` - Clustering comparison
- `unsupervised_dimensionality_reduction_results.png` - PCA and t-SNE results
- `elbow_method_results.png` - Optimal cluster selection

### 3. Reinforcement Learning (`reinforcement_learning_example.py`)

**What it demonstrates:**
- **Q-Learning**: Custom grid world environment
- **Policy Gradient (REINFORCE)**: CartPole environment from OpenAI Gym
- Algorithm comparison and hyperparameter effects

**Key Concepts:**
- Environment interaction
- Reward maximization
- Exploration vs exploitation
- Policy optimization

**Outputs:**
- `q_learning_results.png` - Q-Learning training progress
- `policy_gradient_results.png` - Policy Gradient training progress
- `rl_algorithm_comparison.png` - Algorithm comparison

## 🎯 Learning Objectives

After running these examples, you should understand:

### Supervised Learning
- How to train models on labeled data
- Different types of supervised learning (classification vs regression)
- Model evaluation and interpretation
- Feature importance analysis

### Unsupervised Learning
- How to find patterns in unlabeled data
- Clustering algorithms and their applications
- Dimensionality reduction for visualization and preprocessing
- Methods for determining optimal parameters

### Reinforcement Learning
- How agents learn through environment interaction
- Value-based vs policy-based methods
- Exploration strategies
- Training progress visualization

## 🔧 Technical Details

### Dependencies
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **gym**: Reinforcement learning environments

### Datasets Used
- **Iris Dataset**: Classic classification dataset (4 features, 3 classes)
- **Boston Housing**: Regression dataset (13 features, continuous target)
- **Custom Grid World**: Simple RL environment (5x5 grid with obstacles)
- **CartPole**: Classic RL environment from OpenAI Gym

## 📈 Key Metrics Explained

### Supervised Learning
- **Accuracy**: Percentage of correct predictions (classification)
- **R² Score**: Proportion of variance explained (regression)
- **RMSE**: Root Mean Square Error (regression)

### Unsupervised Learning
- **Silhouette Score**: Measures cluster quality (-1 to 1, higher is better)
- **Adjusted Rand Index**: Compares clustering to true labels (0 to 1, higher is better)
- **Explained Variance**: Proportion of variance captured by PCA components

### Reinforcement Learning
- **Episode Reward**: Total reward accumulated in one episode
- **Episode Length**: Number of steps taken in one episode
- **Average Reward**: Mean reward over multiple episodes

## 🎨 Visualization Features

Each example includes comprehensive visualizations:
- Training progress plots
- Model performance comparisons
- Feature importance analysis
- Data distribution plots
- Algorithm comparison charts

## 🔍 Customization

### Modifying Parameters
You can easily modify hyperparameters in each script:
- Learning rates
- Number of training episodes
- Model architectures
- Environment parameters

### Adding New Algorithms
The code structure makes it easy to add new algorithms:
- Follow the existing class structure
- Implement the required methods
- Add visualization code

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Make sure all dependencies are installed
2. **Memory Issues**: Reduce dataset size or number of episodes
3. **Slow Training**: Reduce number of episodes for quick testing

### Performance Tips
- Use smaller datasets for quick experimentation
- Reduce number of training episodes for faster runs
- Adjust hyperparameters based on your specific needs

## 📚 Further Reading

### Supervised Learning
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

### Unsupervised Learning
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- [Clustering Algorithms Comparison](https://scikit-learn.org/stable/modules/clustering.html)

### Reinforcement Learning
- "Reinforcement Learning: An Introduction" by Sutton and Barto
- [OpenAI Gym Documentation](https://gym.openai.com/)

## 🤝 Contributing

Feel free to contribute by:
- Adding new algorithms
- Improving visualizations
- Adding more datasets
- Enhancing documentation

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Learning! 🎓**

These examples provide a solid foundation for understanding the three main types of machine learning. Experiment with the parameters, try different datasets, and explore the code to deepen your understanding of machine learning concepts.
