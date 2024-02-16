import cipher as cp
import torch
from tqdm import tqdm

class EffectTest():
    def __init__(self, model, test_loader, criterion, device, output_dim):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.output_dim = output_dim
      
    def find_nearest(self, array, value):
        '''Algorithm to find the nearest value in an array.'''
        array = torch.tensor(array).to(self.device)
        idx = (torch.abs(array - value)).argmin()
        return array[idx]
    
    def test(self):
        self.model.eval()
        
        epoch_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing") as pbar:
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

                    pbar.update(1)

        # self.index_analysis() # Waiting to be done in the future
        accuracy = total_accuracy / len(self.test_loader)
        print(f"Test accuracy: {accuracy}")
        return accuracy

    def index_analysis(self):
        '''
        There are many indexes that can judge the statbility and translation efficiency of mRNA.
        Except for Accuracy in the sense of codon degeneracy, we can also use the following indexes:
        CG content, CAI, FOP, GC3s, tAI, tRNA adaptation index, etc.
        But this model is just tring to constuct the mapping from amino acid to mRNA(DNA in fact) as the data we proivided. So this part is waiting to be done in the future.
        '''
        pass