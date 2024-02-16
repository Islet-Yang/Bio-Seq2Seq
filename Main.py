import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import cipher as cp
from MyDataset import MyDataset
from EarlyStopping import EarlyStopping
from EffectTest import EffectTest
from Network.Modules import Transformer


class ModelWorking():
    def __init__(self):
        
        self.args = self.get_args()
        self.print_args()
        
        # Set seed
        self.set_seed()
        
        assert self.args.encoder_emb_dim == self.args.decoder_emb_dim, "Embedding dimensions of encoder and decoder should be equal!"
        assert self.args.d_v == self.args.d_q_and_d_k // self.args.n_heads, "d_k and d_v should be equal to decoder embedding dimension divided by number of heads!"

        self.early_stopping = EarlyStopping(patience=self.args.patience, checkpoint_path=self.args.save_path, mode='max')

        # Create model
        self.model = Transformer(n_src_vocab=self.args.input_dim, n_trg_vocab=self.args.output_dim, src_pad_idx=self.args.pad_idx, trg_pad_idx=self.args.pad_idx, d_word_vec=self.args.encoder_emb_dim, d_model=self.args.decoder_emb_dim, d_inner=self.args.hidden_dim, n_layers=self.args.n_layers, n_head=self.args.n_heads, d_k=self.args.d_q_and_d_k, d_v=self.args.d_v, dropout=self.args.dropout, n_position=self.args.max_length, trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True, scale_emb_or_prj='prj').to(self.args.device)
        
        # Define your loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        # Create data_loader
        self.createloaders()
        
        # Create test object
        self.Test=EffectTest(self.model, self.test_loader, self.criterion, self.args.device, self.args.output_dim)

    def get_args(self):
        '''Get all the parameters from command line'''
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--seed', type=int, default=19, help='random seed')
        parser.add_argument('--device', type=str, default='cuda', help='device')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        parser.add_argument('--pad_idx', type=int, default=0, help='pad index')
        parser.add_argument('--clip', type=int, default=1, help='clip')
        
        parser.add_argument('--max_length', type=int, default=700, help='max length')
        parser.add_argument('--input_dim', type=int, default=21, help='input dimension')
        parser.add_argument('--output_dim', type=int, default=64, help='output dimension')
        parser.add_argument('--encoder_emb_dim', type=int, default=32, help='encoder embedding dimension')
        parser.add_argument('--decoder_emb_dim', type=int, default=32, help='decoder embedding dimension')
        parser.add_argument('--d_v', type=int, default=32, help='d_v')
        parser.add_argument('--d_q_and_d_k', type=int, default=128, help='d_q_and_d_k')
        
        parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')       
        parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
        parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        
        parser.add_argument('--epoch_num', type=int, default=100, help='number of epochs')
        parser.add_argument('--patience', type=int, default=20, help='patience')
        
        parser.add_argument('--aa_file_path', type=str, default='./data/aa_data_nonrepetitive.fasta', help='AA file path')
        parser.add_argument('--dna_file_path', type=str, default='./data/dna_data_nonrepetitive.fasta', help='DNA file path')
        parser.add_argument('--save_path', type=str, default='best_model.pth', help='save path')
        
        return parser.parse_args()
    
    def print_args(self):
        '''Print all the parameters'''
        
        print("*********************** Start Training ***********************\n")
        print("            Device: ", self.args.device)
        print("            Batch size: ", self.args.batch_size)
        print("\n")
        print("            Max length: ", self.args.max_length)
        print("            Input dimension: ", self.args.input_dim)
        print("            Output dimension: ", self.args.output_dim)
        print("            Encoder embedding dimension: ", self.args.encoder_emb_dim)
        print("            Decoder embedding dimension: ", self.args.decoder_emb_dim)
        print("            d_v: ", self.args.d_v)
        print("            d_q_and_d_k: ", self.args.d_q_and_d_k)
        print("            Hidden dimension: ", self.args.hidden_dim)
        print("            Number of layers: ", self.args.n_layers)
        print("            Number of heads: ", self.args.n_heads)
        print("            Dropout: ", self.args.dropout)
        print("\n")
        print("            Learning rate: ", self.args.learning_rate)
        print("            Weight decay: ", self.args.weight_decay)
        print("            Number of epochs: ", self.args.epoch_num)
        print("            Patience: ", self.args.patience)
        print("**************************************************************\n\n\n")

    def set_seed(self):
        '''Set seed for all random cases'''
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def createloaders(self):
        '''Create train, validation and test data loaders'''
        all_dataset = MyDataset(self.args.aa_file_path, self.args.dna_file_path)
        
        # shuffle all data
        indices = list(range(all_dataset.data_size()))
        random.shuffle(indices)

        # divide train/validation/test dataset
        train_ratio = 0.6
        val_ratio = 0.3
        test_ratio = 0.1

        train_size = int(train_ratio * len(indices))
        val_size = int(val_ratio * len(indices))
        test_size = int(test_ratio * len(indices))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]

        train_dataset = Subset(all_dataset, train_indices)
        val_dataset = Subset(all_dataset, val_indices)
        test_dataset = Subset(all_dataset, test_indices)
        
        # Create data_loader
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: {'input': pad_sequence([item['input'] for item in x], batch_first=True, padding_value=0),'target': pad_sequence([item['target'] for item in x], batch_first=True, padding_value=0)})
        self.validation_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: {'input': pad_sequence([item['input'] for item in x], batch_first=True, padding_value=0),'target': pad_sequence([item['target'] for item in x], batch_first=True, padding_value=0)})
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=lambda x: {'input': pad_sequence([item['input'] for item in x], batch_first=True, padding_value=0),'target': pad_sequence([item['target'] for item in x], batch_first=True, padding_value=0)})

    def find_nearest(self, array, value):
        '''Algorithm to find the nearest value in an array.'''
        array = torch.tensor(array).to(self.args.device)
        idx = (torch.abs(array - value)).argmin()
        return array[idx]
      
    def train(self, epoch):
        '''Training'''    
        self.model.train()
        
        epoch_loss = 0
        total_accuracy = 0
        
        with tqdm(total=len(self.train_loader), desc=f'Training[epoch{epoch+1}]', unit='batch') as pbar:
            for i, batch in enumerate(self.train_loader):
              
                src = batch['input'].to(self.args.device)
                trg = batch['target'].to(self.args.device)
                
                #check
                if(len(src)!=len(trg)):
                    pbar.update(1)
                    continue
                
                self.optimizer.zero_grad()            
                output = self.model(src, trg)               
                output_dim = output.shape[-1]
              
                output = output.view(-1, output_dim)
                trg = trg.view(-1)
                
                loss = self.criterion(output, trg)                
                loss.backward()                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)               
                self.optimizer.step()               
                epoch_loss += loss.item()  

                # Convert output to predicted tokens
                # output = output.view(-1, self.output_dim)
                
                pred = output.argmax(1).to(self.args.device)
                
                # Find the nearest value in degeneracy_table
                # pred = [self.find_nearest(cp.degeneracy_table[str(src[i+1].item())], pred[i]) for i in range(len(src)-1)]
                
                pred = torch.stack([self.find_nearest(cp.degeneracy_table[str(s.item())], p) for s, p in zip(src.flatten(), pred)])
                pred = torch.tensor(pred).to(self.args.device)
                
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
        '''Validation'''
        self.model.eval()
        
        epoch_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            with tqdm(total=len(self.validation_loader), desc="Evaluating") as pbar:
                for i, batch in enumerate(self.validation_loader):

                    src = batch['input'].to(self.args.device)
                    trg = batch['target'].to(self.args.device)
                    
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
                    pred = output.argmax(1).to(self.args.device)
                    
                    # Find the nearest value in degeneracy_table
                    pred = torch.stack([self.find_nearest(cp.degeneracy_table[str(s.item())], p) for s, p in zip(src.flatten(), pred)])
                    pred = torch.tensor(pred).to(self.args.device)
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
        '''Testing'''
        self.model.eval()
        
        epoch_loss = 0
        total_accuracy = 0
        all_results = []
        
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing") as pbar:
                for i, batch in enumerate(self.test_loader):
                    src = batch['input'].to(self.args.device)
                    trg = batch['target'].to(self.args.device)
                    
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
                    output = output.view(-1, self.args.output_dim)
                    pred = output.argmax(1).to(self.args.device)
                    
                    # Find the nearest value in degeneracy_table
                    pred = torch.stack([self.find_nearest(cp.degeneracy_table[str(s.item())], p) for s, p in zip(src[1:].flatten(), pred)])
                    pred = torch.tensor(pred).to(self.args.device)
                    
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
        '''Main function'''
        with open('accuracy_record.txt', 'w') as file:
            for epoch in range(self.args.epoch_num):
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
            # test_accuracy = self.test()
            test_accuracy=self.Test.test()
            file.write(f"\nTest accuracy: {test_accuracy}\n")

# Main
if __name__ == '__main__':
    analysis = ModelWorking()
    analysis.main()
