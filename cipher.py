
# Cipher stores the cipher for the DNA and amino acid alphabets.
# degenracy_table stores the degeneracy of each codon. The codons are represneted in a exquisitely designed way.

dna_aa_alphabet = {'AAA':'a','AAT':'b','AAG':'c','AAC':'d','ATA':'e','ATT':'f','ATG':'g','ATC':'h',
                    'AGA':'i','AGT':'j','AGG':'k','AGC':'l','ACA':'m','ACT':'n','ACG':'o','ACC':'p',
                    'TAA':'q','TAT':'r','TAG':'s','TAC':'t','TTA':'u','TTT':'v','TTG':'w','TTC':'x',
                    'TGA':'y','TGT':'z','TGG':'A','TGC':'B','TCA':'C','TCT':'D','TCG':'E','TCC':'F',
                    'GAA':'G','GAT':'H','GAG':'I','GAC':'J','GTA':'K','GTT':'L','GTG':'M','GTC':'N',
                    'GGA':'O','GGT':'P','GGG':'Q','GGC':'R','GCA':'S','GCT':'T','GCG':'U','GCC':'V',
                    'CAA':'W','CAT':'X','CAG':'Y','CAC':'Z','CTA':'!','CTT':'@','CTG':'#','CTC':'$',
                    'CGA':'%','CGT':'^','CGG':'&','CGC':'*','CCA':'(','CCT':')','CCG':'-','CCC':'+'}

codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',  # Phe, Leu
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',  # Leu
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',  # Ile, Met (起始)
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',  # Val
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',  # Ser
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',  # Pro
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',  # Thr
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',  # Ala
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',  # Tyr, 终止
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',  # His, Gln
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',  # Asn, Lys
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',  # Asp, Glu
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',  # Cys, 终止, Trp
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',  # Arg
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',  # Ser, Arg
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',  # Gly
    }

dna_emb = {'A':30,'B':34,'C':46,'D':10,'E':39,'F':2,'G':44,'H':6,
                    'I':26,'J':43,'K':3,'L':36,'M':48,'N':11,'O':41,'P':60,
                    'Q':53,'R':7,'S':25,'T':49,'U':4,'V':56,'W':52,'X':12,
                    'Y':21,'Z':40,'a':62,'b':22,'c':28,'d':0,'e':5,'f':57,
                    'g':42,'h':17,'i':20,'j':31,'k':1,'l':23,'m':29,'n':45,
                    'o':35,'p':58,'q':54,'r':13,'s':16,'t':63,'u':8,'v':24,
                    'w':27,'x':18,'y':37,'z':59,'!':55,'@':14,'#':19,'$':33,
                    '%':51,'^':9,'&':47,'*':32,'(':50,')':61,'-':38,'+':15}

amino_acid_emb = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,
                    'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,
                    'T':16,'V':17,'W':18,'Y':19,'*':20}

degeneracy_table = {'0':[4,25,49,56],'1':[34,59],'2':[6,43],'3':[26,44],'4':[18,24],'5':[7,41,53,60],
                    '6':[12,40],'7':[5,17,57],'8':[28,62],'9':[8,27,55,14,19,33],'10':[42],'11':[0,22],'12':[15,38,50,61],'13':[21,52],'14':[51,9,47,32,1,20],'15':[46,10,39,2,31,23],
                    '16':[29,45,35,58],'17':[3,36,48,11],'18':[30],'19':[13,63],'20':[54,16,37]}

amino_acid_tensortable = {'A':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'C':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'D':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'E':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                            'F':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'G':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'H':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'I':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'K':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            'L':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            'M':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                            'N':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                            'P':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            'Q':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                            'R':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                            'S':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                            'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                            'V':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                            'W':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                            'Y':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                            '*':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
