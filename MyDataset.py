import numpy as np
import cipher as cp
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, aa_file, dna_file):
        self.aa_sequences = self.load_sequences(aa_file)
        self.dna_sequences = self.load_sequences(dna_file)
        assert len(self.aa_sequences) == len(self.dna_sequences), "AA and DNA sequences should have the same length."

    def load_sequences(self, file):
        '''Load sequences from a fasta file.'''
        sequences = []
        with open(file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences.append(str(record.seq))
        return sequences

    def __len__(self):
        return len(self.aa_sequences)

    def __getitem__(self, idx):
        aa_sequence = self.aa_sequences[idx]
        dna_sequence = self.dna_sequences[idx]

        # Encode the sequences
        aa_sequence_indices = [cp.amino_acid_emb[base] for base in aa_sequence]
        dna_sequence_indices = [cp.dna_emb[base] for base in dna_sequence]
        aa_sequence_indices = torch.tensor(aa_sequence_indices)
        dna_sequence_indices = torch.tensor(dna_sequence_indices)

        return {
            'input': aa_sequence_indices,
            'target': dna_sequence_indices
        }
        
    def encode_sequence(self, sequence, symbol_dict):
        '''Encode a sequence using a symbol dictionary.'''
        n = len(sequence)
        num_symbols = len(symbol_dict)
        encoded_sequence = np.zeros((n, num_symbols), dtype=int)

        for i, symbol in enumerate(sequence):
            index = symbol_dict.get(symbol, -1)
            if index != -1:
                encoded_sequence[i, index] = 1

        return torch.tensor(encoded_sequence)
      
    def data_size(self):
        return len(self.aa_sequences)


# Test the dataset
if __name__ == '__main__':
    from torch.nn.utils.rnn import pad_sequence
    
    # Create a dataset
    dataset = MyDataset('./data/aa_data_nonrepetitive.fasta', './data/dna_data_nonrepetitive.fasta')
    print(dataset.data_size())
    
    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: {'input': pad_sequence([item['input'] for item in x], batch_first=True, padding_value=0),'target': pad_sequence([item['target'] for item in x], batch_first=True, padding_value=0)})
    
    # Test the dataloader
    for i,batch in enumerate(dataloader):
        if(i == 1):
            break # Just test the first batch
        
        src = batch['input']
        trg = batch['target']
        print(src)
        print(trg)
    
