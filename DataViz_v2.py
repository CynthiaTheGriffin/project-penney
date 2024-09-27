import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_heatmaps(format:str) -> None:
    ################################
    ## Generate test data - START ##
    ################################
    data = pd.read_csv('Variation1.csv')
    
    # mean() is redundant here
    # This mainly exists for readability later on
    probs = data[['Sequence 1', 'Sequence 2', 'Player 1 Win %']].groupby(['Sequence 1', 'Sequence 2']).mean()
    
    seqs = ['000', '001', '010', '011', '100', '101', '110', '111'] # All possible color sequences
    matrix = np.zeros((8,8)) # For storing probabilities

    # Fill matrix
    for p1_idx, p1_seq in enumerate(seqs):
        for p2_idx, p2_seq in enumerate(seqs):
            if p1_seq == p2_seq:
                # Make duplicate sequnce pairs nan values (to be black on heatmap later)
                matrix[p1_idx][p2_idx] = np.nan
                continue
                
            val = probs.loc[int(p1_seq)].loc[int(p2_seq)].values[0]
            matrix[p1_idx][p2_idx] = round(val)
    ##############################
    ## Generate test data - END ##
    ##############################
    
    #######################
    ## Generate heatmaps ##
    #######################
    N = 0 # Placeholder
    generate_heatmap(matrix, N=N, filename = 'Variation1.'+format)
    #generate_heatmap(other_matrix, N=N, filename = 'Variation2.'+format)
    return
    
def generate_heatmap(matrix, N:int, filename:str) -> None:                
    # Generate heatmap
    # Everything below here is the part that matters
    seqs = ['RRR', 'RRB', 'RBR', 'RBB', 'BRR', 'BRB', 'BBR', 'BBB'] # All possible color sequences (but in letters)
    plt.figure(figsize=(8,8))
    ax = sns.heatmap(matrix, 
                     annot = True, fmt='.0f',  cmap='Blues',
                     vmin = 0, vmax = 100, 
                     cbar_kws={'format':'%.0f%%', 'location':'left'},
                     xticklabels = seqs)
    ax.set_yticklabels(seqs, verticalalignment='center')
    ax.set_xlabel('Player 2 Sequence')
    ax.set_ylabel('Player 1 Sequence', rotation=270)
    ax.yaxis.set_label_coords(1.08, 0.5) # Manually set ylabel position
    ax.yaxis.tick_right() # Move yticks to the right side
    ax.set_facecolor('lightgray') # Make diagonal light gray
    plt.title(f'Player 1 Win Percents Over All Winned Games (N={N})')
    plt.yticks(rotation=270)
        
    # Save heatmap
    ax.get_figure().savefig(filename)
    
    print(f'<> {filename} saved <>')
    return