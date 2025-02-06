import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class WordFrequencyAnalyzer:
    def __init__(self, excel_file):
        """Initialize analyzer with Excel file path"""
        self.df = pd.read_excel(excel_file)
        self.letter_freq = defaultdict(float)
        self.bigram_freq = defaultdict(float)
        self.weighted_letter_freq = defaultdict(float)
        # New: Add positional bigram tracking
        self.word_start_bigrams = defaultdict(float)
        self.word_end_bigrams = defaultdict(float)
        self.same_hand_bigrams = defaultdict(float)
        
        # Define which letters are typed by which hand
        self.left_hand = set('qwertasdfgzxcvb')
        self.right_hand = set('yuiophjklnm')
        
    def analyze_letters(self, word_col='Word', freq_col='Frequency', word_rank_weight=0.8):
        """Analyze letter and bigram frequencies with position awareness"""
        total_words = self.df[freq_col].sum()
        self.df['weight'] = self.df[freq_col] / total_words
        self.df['rank_weight'] = (1 / (self.df.index + 1)) ** word_rank_weight
        
        for idx, row in self.df.iterrows():
            word = str(row[word_col]).lower()
            weight = row['weight'] * row['rank_weight']
            
            # Single letter frequencies
            for letter in word:
                if letter.isalpha():
                    self.weighted_letter_freq[letter] += weight
                    self.letter_freq[letter] += 1
            
            # Enhanced bigram analysis
            self._analyze_bigrams(word, weight)
    
    def _analyze_bigrams(self, word, weight):
        """Detailed bigram analysis including positional and hand-based patterns"""
        if len(word) < 2:
            return
            
        # Regular bigrams with weighting
        for i in range(len(word) - 1):
            if word[i].isalpha() and word[i+1].isalpha():
                bigram = word[i:i+2]
                self.bigram_freq[bigram] += weight
                
                # Track same-hand bigrams
                if (word[i] in self.left_hand and word[i+1] in self.left_hand) or \
                   (word[i] in self.right_hand and word[i+1] in self.right_hand):
                    self.same_hand_bigrams[bigram] += weight
        
        # Start and end bigrams
        if len(word) >= 2:
            if word[0].isalpha() and word[1].isalpha():
                self.word_start_bigrams[word[0:2]] += weight
            if word[-2].isalpha() and word[-1].isalpha():
                self.word_end_bigrams[word[-2:]] += weight
    
    def get_top_letters(self, n=26):
        """Get top n most frequent letters with their weighted frequencies"""
        sorted_letters = sorted(self.weighted_letter_freq.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        return sorted_letters[:n]
    
    def get_bigram_analysis(self, n=20):
        """Get comprehensive bigram analysis"""
        return {
            'all_bigrams': sorted(self.bigram_freq.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:n],
            'start_bigrams': sorted(self.word_start_bigrams.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:n],
            'end_bigrams': sorted(self.word_end_bigrams.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:n],
            'same_hand_bigrams': sorted(self.same_hand_bigrams.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:n]
        }
    
    def plot_letter_frequencies(self):
        """Plot letter frequency distribution"""
        letters, freqs = zip(*self.get_top_letters())
        
        plt.figure(figsize=(15, 6))
        sns.barplot(x=list(letters), y=list(freqs))
        plt.title('Weighted Letter Frequencies')
        plt.xlabel('Letters')
        plt.ylabel('Weighted Frequency')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_bigram_analysis(self):
        """Plot comprehensive bigram analysis"""
        # Create a figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # Helper function for heatmap plotting
        def plot_bigram_heatmap(data, ax, title):
            matrix = np.zeros((26, 26))
            letters = 'abcdefghijklmnopqrstuvwxyz'
            letter_to_idx = {letter: idx for idx, letter in enumerate(letters)}
            
            for bigram, freq in data:
                if len(bigram) == 2 and bigram[0] in letter_to_idx and bigram[1] in letter_to_idx:
                    i, j = letter_to_idx[bigram[0]], letter_to_idx[bigram[1]]
                    matrix[i][j] = freq
            
            sns.heatmap(matrix, xticklabels=list(letters), yticklabels=list(letters),
                       ax=ax, cmap='YlOrRd', annot=False)
            ax.set_title(title)
            ax.set_xlabel('Second Letter')
            ax.set_ylabel('First Letter')
        
        # Plot different types of bigrams
        bigram_analysis = self.get_bigram_analysis()
        plot_bigram_heatmap(bigram_analysis['all_bigrams'], ax1, 'All Bigrams')
        plot_bigram_heatmap(bigram_analysis['start_bigrams'], ax2, 'Word Start Bigrams')
        plot_bigram_heatmap(bigram_analysis['end_bigrams'], ax3, 'Word End Bigrams')
        plot_bigram_heatmap(bigram_analysis['same_hand_bigrams'], ax4, 'Same Hand Bigrams')
        
        plt.tight_layout()
        plt.show()
    
    def export_analysis(self, output_file='keyboard_analysis.txt'):
        """Export comprehensive analysis results to a file"""
        with open(output_file, 'w') as f:
            f.write("=== Keyboard Layout Analysis ===\n\n")
            
            f.write("=== Letter Frequencies ===\n")
            for letter, freq in self.get_top_letters():
                f.write(f"{letter}: {freq:.6f}\n")
            
            f.write("\n=== Bigram Analysis ===\n")
            bigram_analysis = self.get_bigram_analysis()
            
            for category, bigrams in bigram_analysis.items():
                f.write(f"\n--- {category.replace('_', ' ').title()} ---\n")
                for bigram, freq in bigrams:
                    f.write(f"{bigram}: {freq:.6f}\n")
            
            # Add hand alternation statistics
            total_bigrams = sum(self.bigram_freq.values())
            same_hand_total = sum(self.same_hand_bigrams.values())
            hand_alt_ratio = 1 - (same_hand_total / total_bigrams if total_bigrams > 0 else 0)
            
            f.write(f"\n=== Hand Alternation Analysis ===\n")
            f.write(f"Hand alternation ratio: {hand_alt_ratio:.2%}\n")
            f.write(f"Same hand bigram ratio: {1-hand_alt_ratio:.2%}\n")

def generate_keyboard_insights(analyzer):
    """Generate insights about keyboard layout optimization"""
    bigram_analysis = analyzer.get_bigram_analysis()
    top_letters = analyzer.get_top_letters(10)
    
    insights = []
    
    # Analyze most frequent letters
    insights.append("Most frequent letters should be on home row:")
    insights.extend([f"  {letter}: {freq:.4f}" for letter, freq in top_letters[:8]])
    
    # Analyze hand alternation
    same_hand = [(bigram, freq) for bigram, freq in bigram_analysis['same_hand_bigrams'][:5]]
    insights.append("\nMost frequent same-hand combinations (consider separating):")
    insights.extend([f"  {bigram}: {freq:.4f}" for bigram, freq in same_hand])
    
    # Analyze word starts
    insights.append("\nMost common word-starting combinations (should be easily accessible):")
    insights.extend([f"  {bigram}: {freq:.4f}" 
                    for bigram, freq in bigram_analysis['start_bigrams'][:5]])
    
    return "\n".join(insights)

# Example usage
if __name__ == "__main__":
    analyzer = WordFrequencyAnalyzer('wordFrequency.xlsx')
    analyzer.analyze_letters()
    
    # Generate visualizations
    analyzer.plot_letter_frequencies()
    analyzer.plot_bigram_analysis()
    
    # Export detailed analysis
    analyzer.export_analysis()
    
    # Print insights
    print("\nKeyboard Layout Optimization Insights:")
    print(generate_keyboard_insights(analyzer))