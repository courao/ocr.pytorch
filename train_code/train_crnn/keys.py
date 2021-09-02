import pickle as pkl
from pathed import filedir
print(filedir)
alphabet_list = pkl.load(open(filedir / "alphabet.pkl", "rb"))
alphabet = [ord(ch) for ch in alphabet_list]
alphabet_v2 = alphabet
