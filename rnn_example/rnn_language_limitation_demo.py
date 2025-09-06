import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimpleRNN(nn.Module):
    """A simple RNN implementation to demonstrate sequential processing limitations"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        rnn_out, hidden = self.rnn(x)
        # Use the last output for classification
        output = self.fc(rnn_out[:, -1, :])
        return output, hidden

class LanguageRNNDemo:
    """Demonstration of RNN limitations with cross-language sequential dependencies"""
    
    def __init__(self):
        # Vocabulary for our simple language examples
        self.vocab = {
            'the': 0, 'european': 1, 'economic': 2, 'area': 3,
            'la': 4, 'zone': 5, 'europeanne': 6,
            'le': 7, 'marche': 8, 'unique': 9,
            'unified': 10, 'market': 11
        }
        
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Create the RNN model
        self.model = SimpleRNN(
            input_size=self.vocab_size,
            hidden_size=64,
            output_size=2  # Binary classification: correct/incorrect word order
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
    def create_one_hot(self, word_idx: int) -> torch.Tensor:
        """Convert word index to one-hot encoding"""
        one_hot = torch.zeros(self.vocab_size)
        one_hot[word_idx] = 1.0
        return one_hot
    
    def sentence_to_tensor(self, sentence: List[str]) -> torch.Tensor:
        """Convert sentence to tensor of one-hot encodings"""
        tensors = []
        for word in sentence:
            if word in self.vocab:
                tensors.append(self.create_one_hot(self.vocab[word]))
            else:
                # Unknown word - use zero vector
                tensors.append(torch.zeros(self.vocab_size))
        return torch.stack(tensors).unsqueeze(0)  # Add batch dimension
    
    def generate_training_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Generate training data with correct and incorrect word orders"""
        
        # English sentences (correct order)
        english_sentences = [
            ['the', 'european', 'economic', 'area'],
            ['the', 'unified', 'market'],
            ['the', 'economic', 'zone']
        ]
        
        # French sentences (different word order)
        french_sentences = [
            ['la', 'zone', 'economic', 'europeanne'],
            ['le', 'marche', 'unique'],
            ['la', 'zone', 'economic']
        ]
        
        # Incorrect word orders (mixing languages)
        incorrect_sentences = [
            ['the', 'zone', 'economic', 'europeanne'],  # English article, French word order
            ['la', 'european', 'economic', 'area'],     # French article, English word order
            ['the', 'marche', 'unique'],                # English article, French words
            ['le', 'unified', 'market']                 # French article, English words
        ]
        
        X = []
        y = []
        
        # Correct sentences (label 1)
        for sentence in english_sentences + french_sentences:
            X.append(self.sentence_to_tensor(sentence))
            y.append(1)
        
        # Incorrect sentences (label 0)
        for sentence in incorrect_sentences:
            X.append(self.sentence_to_tensor(sentence))
            y.append(0)
        
        return X, y
    
    def train_model(self, epochs: int = 100):
        """Train the RNN model"""
        X, y = self.generate_training_data()
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for i, (sentence_tensor, label) in enumerate(zip(X, y)):
                self.optimizer.zero_grad()
                
                # Forward pass
                output, _ = self.model(sentence_tensor)
                loss = self.criterion(output, torch.tensor([label]))
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.argmax(output).item()
                if predicted == label:
                    correct += 1
                total += 1
            
            avg_loss = total_loss / len(X)
            accuracy = correct / total
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return losses, accuracies
    
    def test_cross_language_generalization(self):
        """Test how well the model generalizes to new cross-language patterns"""
        
        print("\n" + "="*60)
        print("CROSS-LANGUAGE GENERALIZATION TEST")
        print("="*60)
        
        # Test cases that the model hasn't seen during training
        test_cases = [
            # New English-French mixed patterns
            (['the', 'marche', 'european'], "English article + French word order"),
            (['la', 'market', 'europeanne'], "French article + English word order"),
            (['le', 'economic', 'zone'], "French article + mixed word order"),
            (['the', 'zone', 'unified'], "English article + French-English mix"),
            
            # Pure language sentences (should be correct)
            (['the', 'european', 'market'], "Pure English"),
            (['la', 'zone', 'europeanne'], "Pure French"),
        ]
        
        self.model.eval()
        with torch.no_grad():
            for sentence, description in test_cases:
                sentence_tensor = self.sentence_to_tensor(sentence)
                output, _ = self.model(sentence_tensor)
                probability = torch.softmax(output, dim=1)
                predicted = torch.argmax(output).item()
                
                print(f"\nSentence: {' '.join(sentence)}")
                print(f"Description: {description}")
                print(f"Predicted: {'Correct' if predicted == 1 else 'Incorrect'}")
                print(f"Confidence: {probability[0][predicted].item():.4f}")
    
    def demonstrate_sequential_failure(self):
        """Demonstrate specific failure cases with the user's example"""
        
        print("\n" + "="*60)
        print("SPECIFIC SEQUENTIAL FAILURE DEMONSTRATION")
        print("="*60)
        
        # User's specific example
        english_phrase = ['the', 'european', 'economic', 'area']
        french_phrase = ['la', 'zone', 'economic', 'europeanne']
        
        print(f"\nEnglish: {' '.join(english_phrase)}")
        print(f"French:  {' '.join(french_phrase)}")
        print("\nKey observation: Word order changes significantly!")
        print("- English: Article + Adjective + Adjective + Noun")
        print("- French:  Article + Noun + Adjective + Adjective")
        
        # Show how RNN processes these differently
        self.model.eval()
        with torch.no_grad():
            # Test English phrase
            eng_tensor = self.sentence_to_tensor(english_phrase)
            eng_output, eng_hidden = self.model(eng_tensor)
            eng_prob = torch.softmax(eng_output, dim=1)
            
            # Test French phrase
            fr_tensor = self.sentence_to_tensor(french_phrase)
            fr_output, fr_hidden = self.model(fr_tensor)
            fr_prob = torch.softmax(fr_output, dim=1)
            
            print(f"\nRNN Predictions:")
            print(f"English phrase: {'Correct' if torch.argmax(eng_output).item() == 1 else 'Incorrect'} "
                  f"(confidence: {eng_prob[0][torch.argmax(eng_output)].item():.4f})")
            print(f"French phrase:  {'Correct' if torch.argmax(fr_output).item() == 1 else 'Incorrect'} "
                  f"(confidence: {fr_prob[0][torch.argmax(fr_output)].item():.4f})")
            
            # Show hidden states to demonstrate sequential processing
            print(f"\nHidden state analysis:")
            print(f"English final hidden state norm: {torch.norm(eng_hidden).item():.4f}")
            print(f"French final hidden state norm:  {torch.norm(fr_hidden).item():.4f}")
            
            # Calculate similarity between hidden states
            similarity = torch.cosine_similarity(eng_hidden.flatten(), fr_hidden.flatten(), dim=0)
            print(f"Hidden state similarity: {similarity.item():.4f}")
    
    def plot_training_results(self, losses, accuracies):
        """Plot training results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(accuracies)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('rnn_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main demonstration function"""
    print("RNN Cross-Language Sequential Dependency Demonstration")
    print("=" * 60)
    print("\nThis demo shows how RNNs struggle with sequential patterns")
    print("that vary across languages, particularly word order differences.")
    
    # Create and train the demo
    demo = LanguageRNNDemo()
    
    print("\nTraining RNN model...")
    losses, accuracies = demo.train_model(epochs=100)
    
    # Demonstrate the limitations
    demo.demonstrate_sequential_failure()
    demo.test_cross_language_generalization()
    
    # Plot results
    demo.plot_training_results(losses, accuracies)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. RNNs process sequences sequentially, building up context")
    print("2. When word order changes significantly (like English vs French),")
    print("   the sequential patterns the RNN learned may not generalize")
    print("3. The model may struggle with mixed-language patterns")
    print("4. This demonstrates why attention mechanisms and transformers")
    print("   are more effective for cross-language understanding")

if __name__ == "__main__":
    main()
