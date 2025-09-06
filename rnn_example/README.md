# RNN Cross-Language Sequential Dependency Demonstration

This project demonstrates how Recurrent Neural Networks (RNNs) struggle with sequential patterns that vary across languages, using your specific example of English vs French word order differences.

## The Problem

RNNs process information sequentially, which creates limitations when dealing with languages that have different word orders. For example:

- **English**: "the european economic area" (Article + Adjective + Adjective + Noun)
- **French**: "la zone economic europeanne" (Article + Noun + Adjective + Adjective)

The sequential nature of RNNs makes it difficult for them to generalize across these different word order patterns.

## Files in This Project

- `rnn_language_limitation_demo.py` - Main demonstration code
- `run_demo.py` - Simple runner script with dependency checking
- `requirements.txt` - Python dependencies
- `RNN_Limitations_Explanation.md` - Detailed explanation of the concepts
- `README.md` - This file

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demonstration**:
   ```bash
   python run_demo.py
   ```

   Or directly:
   ```bash
   python rnn_language_limitation_demo.py
   ```

## What the Demonstration Shows

1. **Training Process**: The RNN learns to classify sentences as having correct or incorrect word order
2. **Cross-Language Testing**: Tests how well the model generalizes to new language patterns
3. **Your Specific Example**: Detailed analysis of "the european economic area" vs "la zone economic europeanne"
4. **Hidden State Analysis**: Shows how the RNN's internal representations differ between languages
5. **Visualization**: Training plots showing loss and accuracy over time

## Key Insights

- RNNs develop biases toward the sequential patterns they see during training
- When word order changes significantly between languages, RNNs struggle to generalize
- This limitation is one reason why attention-based models (transformers) are more effective for cross-language tasks
- The demonstration shows why modern NLP systems use more sophisticated architectures

## Expected Output

The demonstration will show:
- Training progress with loss and accuracy metrics
- Analysis of your English/French example
- Generalization tests on unseen cross-language patterns
- Hidden state comparisons between different language structures
- Training plots saved as `rnn_training_results.png`

## Technical Details

- **Framework**: PyTorch
- **Model**: Simple RNN with one-hot word encodings
- **Task**: Binary classification (correct vs incorrect word order)
- **Languages**: English and French with different word order patterns
- **Visualization**: Matplotlib and Seaborn

## Further Exploration

You can extend this demonstration by:
- Adding more languages with different word orders
- Testing with LSTM or GRU variants
- Implementing attention mechanisms for comparison
- Using more complex sentence structures
- Adding position encodings to see the difference

## Why This Matters

This demonstration illustrates fundamental limitations in sequential neural networks that led to the development of:
- Attention mechanisms
- Transformer architectures
- Position encodings
- Modern machine translation systems

Understanding these limitations helps explain why modern NLP systems use more sophisticated architectures than simple RNNs.
