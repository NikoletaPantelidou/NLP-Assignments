
# Import of all necessary libraries and IPython requests to run the code.

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import requests

# download the names.txt file from github
url = "https://raw.githubusercontent.com/NikoletaPantelidou/NLP-Assignments/refs/heads/main/names_greek.txt"  # Replace with the actual URL
response = requests.get(url)

if response.status_code == 200:
    # I save the content as a file
    with open('names_greek.txt', 'wb') as file:
        file.write(response.content)
    print("File downloaded and saved successfully.")
else:
    print("Failed to download file:", response.status_code)

# I open and process the file
with open('names_greek.txt', 'r') as f:
    words = f.read().splitlines()
    
words= [w.lower() for w in words] #I lowercase the words because later the stoi dictionary gets messy.
print(words)

len(words)

"""First change to Andrej's code for non-latin script
(This is a general description, details on the methods are commented below on the code, the exact point of change, is written as 'CHANGE HERE')

The first problem that occurs is the dictionary creation from string to integers. To accurately map a to 1 (α : 1), b to 2 (β:2) until the last letter (ω: 24) as Andrej does, we need to remove accents and other diacritics. To achieve this, I create a funciton called "normalize", which takes the data (text) as argument. This function returns the data in string form after executing the condition statement.
In the condition statement, I apply normalization.It is a method from the unicodedata library that normalizes the data. Passing the NFKD, separates the letters from the accents and other diacritics like umlaut. Subsequently, I create a condition where I keep only the letter characters and not diacritics since the normalization before decomposed accented characters in two strings.
Having my function done, I apply it to my data by iterating it and saving it as a list.
The last change I apply is to filter our the empty strings that appeared in the output and the charcter "ς" in Greek, which cannot be included in the dictionary since it never occurs as first character in a word.

The rest of the code is the same.
"""

# build the vocabulary of characters and mappings to/from integers

import unicodedata #I import the library I will use later

#I create a function called 'normalize' which return as a string the characters (c) without diacritics.
#To achieve this, I first use the unicodedata.normalize() method, where I pass two arguments:
#the text, which corresponds to my data, and the NFKD, which is the normalization form. In this case, I use NFKD because I want to remove accents
#The function returns in a string form only to those characters (letters in this case) that don't belong to the Mn category that includes the diacritics.
def normalize(text): #CHANGE HERE
    #Removes accents and other diacritics from Greek characters.
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')

# Normalize all names
words = [normalize(w) for w in words] #CHANGE HERE: I iterate to my data(words)and normalize them using the function I created above. The rest of the code is from Andrej

# Build the stoi dictionary after normalization
chars = sorted(list(set(''.join(words))))
# Filter out space and 'ς'
chars = [c for c in chars if c != ' ' and c != 'ς'] #CHANGE HERE: This line is added because in the output there was first an empty string in the dicitonary and secind the letter 'ς' is treated like the other letters but it shouldn't be included since it only occurs as a final letter and in no case a word can start with it.
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0  # Special token for end of name
itos = {i: s for s, i in stoi.items()}

print(stoi)
print(itos)

"""Second change to Andrej's code for non-latin script
(The exact point of change of the code is writen as 'CHANGE HERE')

In this stage, I added a slight change. When the for loop starts iterating in the stoi dictionary, 
I added the condition that if a specific character is not included in the stoi dictionary, it can skip it and continue.
This change was added since the letter 'ς' in Greek is used only as final letter and it is not included in the dictionary above. 
In that way, the modelcan continue by ignoring this case.
"""

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []                     #X list is the inputs list and Y is the labels/targets list of characters, the prediction character.
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      if ch not in stoi:  # CHANGE HERE: I got the error 'ς' in the output. So, I skip characters that are not in stoi (e.g., 'ς') in that way it doesn't stop there.
                continue
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)                 #We create tensors instead of lists for X and Y.
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647) # for reproducibility #Manual seed is just a random number that we give, like a key that the model is based on it
C = torch.randn((27, 10), generator=g) #g is a function that we pass to the generator and torch.randn generates random numbers using the generator object we seed on.
W1 = torch.randn((30, 200), generator=g) #30 are the inputs and 200 are the neurons
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

sum(p.nelement() for p in parameters) # number of parameters in total

for p in parameters:
  p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

for i in range(200000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))

  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 2)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

print(loss.item())

plt.plot(stepi, lossi)

plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
plt.show()

# training loss
#We need to run everything to get the model's predictions
emb = C[Xtr] # (32, 3, 2) #XTR are the rows that corresponds to the input, the training data
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)# We take the emb multiplying them by weights adding bias  and applying tahn
logits = h @ W2 + b2 # (32, 27) # The raw output is the h (first layer)and the weight
train_loss = F.cross_entropy(logits, Ytr) #negative log likehood is different from cross_entropy. This is for distance distribution. The logits is the output we have before applying the sofrmax. They are unormalized counts
print("Training Loss:", train_loss.item()) #Ytr is the next character following

# validation loss
emb = C[Xdev] # (32, 3, 2) #Xdv is the development data
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
val_loss = F.cross_entropy(logits, Ydev)
print("Validation Loss:", val_loss.item())

# test loss
emb = C[Xte] # (32, 3, 2)Xte is the test
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
test_loss = F.cross_entropy(logits, Yte)
print("Test Loss:", test_loss.item())

"""Third change to Andrej's code for non-latin script
(This is a general description more explicit comments are included in the code below, the exact point of the change, is writen as 'CHANGE HERE')

In the following code, the error before applying the changes was just a number (25).
I suspected that since the latin alphabet has 26 characters, in the case of greek alphabet that is not the case since it has 24. 
During the iteration in the last index itos[25], probably it didn't find any character and gave the error 25.
To fix this, I explicitly defined the character index being equal to 24 (length of itos(which is 25) - 1) and also setting the else as 0 to handle the '.' character that has the position 0.
"""

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
#Ι added the line that follows below because the original code expected 26 characters (like the Latin alphabet: A-Z)+ the special dot, but Greek has only 24 letters (Α-Ω). So, in
#my code, i + 1 starts from α which is the first character and goes until the len(itos)which is 25 but we add -1 to stop the iteration to the last letter, ω, which is the position 24.
#Else, 0 is the special dot character.
    char_index = i + 1 if i < len(itos) - 1 else 0 #CHANGE HERE
    plt.text(C[i,0].item(), C[i,1].item(), itos[char_index], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(itos[i] for i in out))