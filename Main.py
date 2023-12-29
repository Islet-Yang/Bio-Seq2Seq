import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import cipher as cp
from MyDataset import Seq2SeqDataset
from EarlyStopping import EarlyStopping
from Model import Encoder, Decoder, Seq2Seq
from Attention import AEncoder, ADecoder, ASeq2Seq


class ModelWorking():
    def __init__(self):
        self.seed = 19
        self.set_seed()
        
        # self.aa_file_path = './data/aa_data.fasta'
        # self.dna_file_path = './data/dna_data.fasta'
        self.aa_file_path = './data/aa_data_nonrepetitive.fasta'
        self.dna_file_path = './data/dna_data_nonrepetitive.fasta'
        
        self.save_path = 'best_model.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {self.device} device.")
        
        self.batch_size = 64
        self.max_length = 700
        self.input_dim = 21  # ACDEFGHIKLMNPQRSTVWY*
        self.output_dim = 64  # AGTC ** 3
        self.encoder_emb_dim = 128
        self.decoder_emb_dim = 128
        self.hidden_dim = 256
        self.pf_dim = 256
        self.n_layers = 3
        self.n_heads = 8
        self.dropout = 0.1
        
        self.learning_rate = 8e-3
        self.weight_decay = 1e-5
        self.epoch_num = 20
        self.clip = 1
        self.patience = 5

        self.encoder = Encoder(self.input_dim, self.encoder_emb_dim, self.hidden_dim, self.n_layers,self.dropout).to(self.device)
        self.decoder = Decoder(self.output_dim, self.decoder_emb_dim, self.hidden_dim, self.n_layers,  self.dropout).to(self.device)
        self.model = Seq2Seq(self.encoder, self.decoder, self.device).to(self.device)
        self.early_stopping = EarlyStopping(patience=self.patience, checkpoint_path=self.save_path, mode='max')

        # self.aencoder = AEncoder(self.input_dim, self.encoder_emb_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)
        # self.adecoder = ADecoder(self.output_dim, self.decoder_emb_dim, self.hidden_dim, self.n_layers, self.dropout, self.n_heads).to(self.device)
        # self.model = ASeq2Seq(self.aencoder, self.adecoder, self.device).to(self.device)
        
        # Define your loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        all_dataset = Seq2SeqDataset(self.aa_file_path, self.dna_file_path)
        
        # shuffle all data
        indices = list(range(all_dataset.data_size()))
        random.shuffle(indices)

        # divide train/validation/test dataset
        self.train_ratio = 0.06
        self.val_ratio = 0.02
        self.test_ratio = 0.01

        train_size = int(self.train_ratio * len(indices))
        val_size = int(self.val_ratio * len(indices))
        test_size = int(self.test_ratio * len(indices))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]

        train_dataset = Subset(all_dataset, train_indices)
        val_dataset = Subset(all_dataset, val_indices)
        test_dataset = Subset(all_dataset, test_indices)
        
        # Create data_loader
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: {'input': pad_sequence([item['input'] for item in x], batch_first=True, padding_value=0),'target': pad_sequence([item['target'] for item in x], batch_first=True, padding_value=0)})
        self.validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: {'input': pad_sequence([item['input'] for item in x], batch_first=True, padding_value=0),'target': pad_sequence([item['target'] for item in x], batch_first=True, padding_value=0)})
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: {'input': pad_sequence([item['input'] for item in x], batch_first=True, padding_value=0),'target': pad_sequence([item['target'] for item in x], batch_first=True, padding_value=0)})
        # self.validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=-1))
        # self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=-1))

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

    # def find_nearest(self, array, value):
    #     array = torch.tensor(array).to(self.device)
    #     idx = (torch.abs(array - value)).argmin()
    #     #idx = (np.abs(array - value)).argmin()
    #     return array[idx]

    def find_nearest(self, array, value):
        array = torch.tensor(array).to(self.device)
        idx = (torch.abs(array - value)).argmin()
        return array[idx]
      
    def train(self, epoch):    
        self.model.train()
        
        epoch_loss = 0
        total_accuracy = 0
        
        with tqdm(total=len(self.train_loader), desc=f'Training[epoch{epoch+1}]', unit='batch') as pbar:
            for i, batch in enumerate(self.train_loader):
                # batch = pack_padded_sequence(padded_batch, lengths=[len(seq) for seq in padded_batch], batch_first=True, enforce_sorted=False)
                src = batch['input'].transpose(0,1).to(self.device)
                trg = batch['target'].transpose(0,1).to(self.device)
                
                #check
                if(len(src)!=len(trg)):
                    pbar.update(1)
                    continue
                
                self.optimizer.zero_grad() 
                #print("src:",src.shape,"\ntrg",trg.shape,"\n")  #############             
                output = self.model(src, trg)               
                output_dim = output.shape[-1]                
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)
                loss = self.criterion(output, trg)               
                loss.backward()                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)               
                self.optimizer.step()               
                epoch_loss += loss.item()  

                # Convert output to predicted tokens
                output = output.view(-1, self.output_dim)
                pred = output.argmax(1).to(self.device)
                
                # Find the nearest value in degeneracy_table
                # pred = [self.find_nearest(cp.degeneracy_table[str(src[i+1].item())], pred[i]) for i in range(len(src)-1)]
                
                pred = torch.stack([self.find_nearest(cp.degeneracy_table[str(s.item())], p) for s, p in zip(src[1:].flatten(), pred)])
                pred = torch.tensor(pred).to(self.device)
                
                # Calculate accuracy
                correct = (pred == trg).float()
                mask = (trg != 0).float()  # create a mask to ignore padding tokens
                accuracy = (correct * mask).sum() / mask.sum()  # calculate accuracy for each sequence
                total_accuracy += accuracy.item()
                                  
                pbar.update(1)
        
        accuracy = total_accuracy / len(self.train_loader)
        total_loss = epoch_loss / len(self.train_loader)
        print(f"Training accuracy: {accuracy}\nTraining loss: {total_loss}")
        return total_loss, accuracy

    def evaluate(self):
        self.model.eval()
        
        epoch_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            with tqdm(total=len(self.validation_loader), desc="Evaluating") as pbar:
                for i, batch in enumerate(self.validation_loader):
                    src = batch['input'].transpose(0, 1).to(self.device)
                    trg = batch['target'].transpose(0, 1).to(self.device)
                    
                    #check
                    if(len(src)!=len(trg)):
                        pbar.update(1)
                        continue
                      
                    output = self.model(src, trg, 0)
                    output_dim = output.shape[-1]
                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].reshape(-1)
                    loss = self.criterion(output, trg)
                    epoch_loss += loss.item()
                    
                    # Convert output to predicted tokens
                    output = output.view(-1, self.output_dim)
                    pred = output.argmax(1).to(self.device)
                    # Find the nearest value in degeneracy_table
                    #pred = [self.find_nearest(cp.degeneracy_table[str(src[i+1].item())], pred[i]) for i in range(len(src)-1)]
                    pred = torch.stack([self.find_nearest(cp.degeneracy_table[str(s.item())], p) for s, p in zip(src[1:].flatten(), pred)])
                    pred = torch.tensor(pred).to(self.device)
                    # Calculate accuracy

                    correct = (pred == trg).float()
                    mask = (trg != 0).float()  # create a mask to ignore padding tokens
                    accuracy = (correct * mask).sum() / mask.sum()  # calculate accuracy for each sequence
                    total_accuracy += accuracy.item()

                    pbar.update(1)

        accuracy = total_accuracy / len(self.validation_loader)
        total_loss = epoch_loss / len(self.validation_loader)
        print(f"Validation accuracy: {accuracy}\nValidation loss: {total_loss}")
        return total_loss, accuracy

    def test(self):
        self.model.eval()
        
        epoch_loss = 0
        total_accuracy = 0
        all_results = []
        
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing") as pbar:
                for i, batch in enumerate(self.test_loader):
                    src = batch['input'].transpose(0, 1).to(self.device)
                    trg = batch['target'].transpose(0, 1).to(self.device)
                    
                    #check
                    if(len(src)!=len(trg)):
                        pbar.update(1)
                        continue
                      
                    output = self.model(src, trg, 0)
                    output_dim = output.shape[-1]
                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].reshape(-1)
                    loss = self.criterion(output, trg)
                    epoch_loss += loss.item()
                    
                    # Convert output to predicted tokens
                    output = output.view(-1, self.output_dim)
                    pred = output.argmax(1).to(self.device)
                    # Find the nearest value in degeneracy_table
                    # pred = [self.find_nearest(cp.degeneracy_table[str(src[i+1].item())], pred[i]) for i in range(len(src)-1)]
                    pred = torch.stack([self.find_nearest(cp.degeneracy_table[str(s.item())], p) for s, p in zip(src[1:].flatten(), pred)])
                    pred = torch.tensor(pred).to(self.device)
                    # Calculate accuracy

                    correct = (pred == trg).float()
                    mask = (trg != 0).float()  # create a mask to ignore padding tokens
                    accuracy = (correct * mask).sum() / mask.sum()  # calculate accuracy for each sequence
                    total_accuracy += accuracy.item()

                    result = f"Target: {trg}\nPrediction: {pred}\nAccuracy: {accuracy.item()}\n\n"
                    all_results.append(result)
                    pbar.update(1)

        # Write all results to file
        with open('test_results.txt', 'w') as file:
            for result in all_results:
                file.write(result)
        
        accuracy = total_accuracy / len(self.test_loader)
        print(f"Test accuracy: {accuracy}")
        return accuracy

    def main(self):
        with open('accuracy_record.txt', 'w') as file:
            for epoch in range(self.epoch_num):
                _, train_accuracy = self.train(epoch)
                _, validation_accuracy = self.evaluate()
                
                file.write(f"Epoch: {epoch+1}\n")
                file.write(f"Training accuracy: {train_accuracy}, Validation accuracy: {validation_accuracy}\n")
                
                self.early_stopping.step(validation_accuracy, self.model)
                if self.early_stopping.should_stop():
                    print(f"Early stopping, on epoch: {epoch}.")
                    break
                    
        with open('accuracy_record.txt', 'a') as file:
            self.early_stopping.load_checkpoint(self.model)
            test_accuracy = self.test()
            file.write(f"\nTest accuracy: {test_accuracy}\n")

# Main
if __name__ == '__main__':
    analysis = ModelWorking()
    analysis.main()
