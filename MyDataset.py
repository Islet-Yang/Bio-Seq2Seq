import numpy as np
import cipher as cp
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader

class Seq2SeqDataset(Dataset):
    def __init__(self, aa_file, dna_file):
        self.aa_sequences = self.load_sequences(aa_file)
        self.dna_sequences = self.load_sequences(dna_file)
        assert len(self.aa_sequences) == len(self.dna_sequences), "AA and DNA sequences should have the same length."

    def load_sequences(self, file):
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

        aa_sequence_indices = [cp.amino_acid_emb[base] for base in aa_sequence]
        dna_sequence_indices = [cp.dna_emb[base] for base in dna_sequence]
        aa_sequence_indices = torch.tensor(aa_sequence_indices)
        dna_sequence_indices = torch.tensor(dna_sequence_indices)

        return {
            'input': aa_sequence_indices,
            'target': dna_sequence_indices
        }
        
    def encode_sequence(self, sequence, symbol_dict):
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



if __name__ == '__main__':
    dataset = Seq2SeqDataset('./data/aa_data.fasta', './data/dna_data.fasta')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(dataset.data_size())
    for batch in dataloader[0: 10]:
        print(batch['input'].shape)
        print(batch['target'].shape)
