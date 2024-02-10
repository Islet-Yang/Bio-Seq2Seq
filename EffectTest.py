import cipher as cp
import torch
from tqdm import tqdm

class Effect_Test():
    def __init__(self, model, test_loader, criterion, device, output_dim):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.output_dim = output_dim
      
    def find_nearest(self, array, value):
        array = torch.tensor(array).to(self.device)
        idx = (torch.abs(array - value)).argmin()
        return array[idx]
    
    def test(self):
        self.model.eval()
        
        epoch_loss = 0
        total_accuracy = 0
        trg_results = []
        pred_results = []
        
        # Create reverse mapping for dna_emb
        reverse_dna_emb = {v: k for k, v in cp.dna_emb.items()}
        reverse_dna_aa_alphabet = {v: k for k, v in cp.dna_aa_alphabet.items()}
        
        # Open files for writing
        
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing") as pbar:
              with open('random_dna_sequences.txt', 'w') as random_file, open('generated_dna_sequences.txt', 'w') as generated_file:
                  for i, batch in enumerate(self.test_loader):
                      src = batch['input'].to(self.device)
                      trg = batch['target'].to(self.device)
                      
                      #check
                      if(len(src)!=len(trg)):
                          pbar.update(1)
                          continue
                        
                      output = self.model(src, trg)
                      output_dim = output.shape[-1]
                      output = output.view(-1, output_dim)
                      trg = trg.reshape(-1)
                      loss = self.criterion(output, trg)
                      epoch_loss += loss.item()
                      
                      # Convert output to predicted tokens
                      output = output.view(-1, self.output_dim)
                      pred = output.argmax(1).to(self.device)
                      # Find the nearest value in degeneracy_table
                      pred = torch.stack([self.find_nearest(cp.degeneracy_table[str(s.item())], p) for s, p in zip(src.flatten(), pred)])
                      pred = torch.tensor(pred).to(self.device)
                      # Calculate accuracy

                      correct = (pred == trg).float()
                      mask = (trg != 0).float()  # create a mask to ignore padding tokens
                      accuracy = (correct * mask).sum() / mask.sum()  # calculate accuracy for each sequence
                      total_accuracy += accuracy.item()
                      
                      trg_seq = ''.join([reverse_dna_aa_alphabet[reverse_dna_emb[i.item()]] for i in trg]).rstrip('\n')
                      pred_seq = ''.join([reverse_dna_aa_alphabet[reverse_dna_emb[i.item()]] for i in pred]).rstrip('\n')

                      random_file.write(trg_seq)
                      random_file.write('\n')
                      generated_file.write(pred_seq)
                      generated_file.write('\n')              
                    
                      pbar.update(1)


        accuracy = total_accuracy / len(self.test_loader)
        print(f"Test accuracy: {accuracy}")
        return accuracy