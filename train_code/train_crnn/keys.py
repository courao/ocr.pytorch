import pickle as pkl
from getpaths import getpath

file_dir = getpath()
alphabet_list = pkl.load(open(file_dir / "alphabet.pkl", "rb"))
alphabet = [ord(ch) for ch in alphabet_list]
alphabet_v2 = alphabet
