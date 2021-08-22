import pickle as pkl

alphabet_list = pkl.load(open("alphabet.pkl", "rb"))
alphabet = [ord(ch) for ch in alphabet_list]
alphabet_v2 = alphabet
