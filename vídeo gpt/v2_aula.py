import torch
import torch.nn as nn
from torch.nn import functional as F

#hiperparâmetros
batch_size = 16 #tamanho do lote para treinamento paralelo
block_size = 32 #comprimento máximo da sequência de contexto
max_iters = 5000 #número total de iterações de treinamento
eval_interval = 100 #frequência de avaliação durante treinamento
learning_rate = 1e-3 #taxa de aprendizado do otimizador
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
eval_iters = 200 #número de iterações para avaliação de perda
n_embd = 64 #dimensão dos embeddings
n_head = 4 #número de cabeças de atenção
n_layer = 4 #número de camadas transformer
dropout = 0.0 #taxa de dropout para regularização


torch.manual_seed(1337) 
#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() #carregar texto completo do arquivo

#aqui estão todos os caracteres únicos que ocorrem neste texto
chars = sorted(list(set(text))) #extrair e ordenar caracteres únicos
vocab_size = len(chars) #tamanho do vocabulário
#criar um mapeamento de caracteres para inteiros
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #recebe uma string, retorna uma lista de inteiros
decode = lambda l: ''.join([itos[i] for i in l]) #recebe uma lista de inteiros, retorna uma string

#divisão dos dados de treino e teste
data = torch.tensor(encode(text), dtype=torch.long) #converter texto para tensor de inteiros
n = int(0.9*len(data)) #primeiros 90% serão treino, resto validação
train_data = data[:n] #dados de treinamento
val_data = data[n:] #dados de validação

#carregamento de dados
def get_batch(split):
    #gerar um pequeno lote de dados com entradas x e alvos y
    data = train_data if split == 'train' else val_data #escolher conjunto de dados
    ix = torch.randint(len(data) - block_size, (batch_size,)) #índices aleatórios
    x = torch.stack([data[i:i+block_size] for i in ix]) #sequências de entrada
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #alvos deslocados por 1
    x, y = x.to(device), y.to(device) #mover para dispositivo
    return x, y

@torch.no_grad() #desabilitar gradientes para avaliação
def estimate_loss():
    out = {} #dicionário para armazenar perdas
    model.eval() #modo de avaliação
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) #tensor para armazenar perdas
        for k in range(eval_iters):
            X, Y = get_batch(split) #obter lote de dados
            logits, loss = model(X, Y) #forward pass
            losses[k] = loss.item() #armazenar perda
        out[split] = losses.mean() #calcular perda média
    model.train() #voltar ao modo de treinamento
    return out

class Head(nn.Module):
    """uma cabeça de auto-atenção"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) #projeção para chaves
        self.query = nn.Linear(n_embd, head_size, bias=False) #projeção para consultas
        self.value = nn.Linear(n_embd, head_size, bias=False) #projeção para valores
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #máscara triangular

        self.dropout = nn.Dropout(dropout) #camada de dropout

    def forward(self, x):
        B,T,C = x.shape #dimensões do tensor de entrada
        k = self.key(x)   #calcular chaves
        q = self.query(x) #calcular consultas
        #calcular pontuações de atenção 
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #aplicar máscara causal
        wei = F.softmax(wei, dim=-1) #normalizar pontuações
        wei = self.dropout(wei) #aplicar dropout
        #realizar a agregação ponderada dos valores
        v = self.value(x) #calcular valores
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #lista de cabeças
        self.proj = nn.Linear(n_embd, n_embd) #projeção de saída
        self.dropout = nn.Dropout(dropout) #dropout para regularização

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenar saídas das cabeças
        out = self.dropout(self.proj(out)) #projetar e aplicar dropout
        return out

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #expansão 4x
            nn.ReLU(), #ativação não-linear
            nn.Linear(4 * n_embd, n_embd), #contração de volta
            nn.Dropout(dropout), #regularização
        )

    def forward(self, x):
        return self.net(x) #aplicar rede feed-forward

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head #tamanho de cada cabeça
        self.sa = MultiHeadAttention(n_head, head_size) #camada de auto-atenção
        self.ffwd = FeedFoward(n_embd) #camada feed-forward
        self.ln1 = nn.LayerNorm(n_embd) #normalização antes da atenção
        self.ln2 = nn.LayerNorm(n_embd) #normalização antes do feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #conexão residual + atenção
        x = x + self.ffwd(self.ln2(x)) #conexão residual + feed-forward
        return x

#modelo bigram super simples
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #cada token lê diretamente os logits para o próximo token de uma tabela de consulta
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #embeddings de tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #embeddings posicionais
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #blocos transformer
        self.ln_f = nn.LayerNorm(n_embd) #normalização da camada final
        self.lm_head = nn.Linear(n_embd, vocab_size) #cabeça de modelagem de linguagem

    def forward(self, idx, targets=None):
        B, T = idx.shape #dimensões do tensor de entrada

        tok_emb = self.token_embedding_table(idx) #embeddings de tokens
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #embeddings posicionais
        x = tok_emb + pos_emb #somar embeddings
        x = self.blocks(x) #aplicar blocos transformer
        x = self.ln_f(x) #normalização final
        logits = self.lm_head(x) #obter logits

        if targets is None:
            loss = None #sem cálculo de perda durante geração
        else:
            B, T, C = logits.shape #dimensões dos logits
            logits = logits.view(B*T, C) #achatar para cálculo de perda
            targets = targets.view(B*T) #achatar targets
            loss = F.cross_entropy(logits, targets) #calcular perda cross-entropy

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx é um array (B, T) de índices no contexto atual
        for _ in range(max_new_tokens):
            #cortar idx para os últimos block_size tokens
            idx_cond = idx[:, -block_size:]
            #obter as predições
            logits, loss = self(idx_cond)
            #focar apenas no último passo de tempo
            logits = logits[:, -1, :] #torna-se (B, C)
            #aplicar softmax para obter probabilidades
            probs = F.softmax(logits, dim=-1) #(B, C)
            #amostrar da distribuição
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #anexar índice amostrado à sequência em execução
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx

model = BigramLanguageModel() #instanciar modelo
m = model.to(device) #mover modelo para dispositivo
#imprimir o número de parâmetros no modelo
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#criar um otimizador PyTorch
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters): #loop de treinamento

    #de vez em quando avaliar a perda nos conjuntos de treino e validação
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss() #calcular perdas
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #amostrar um lote de dados
    xb, yb = get_batch('train')

    #avaliar a perda
    logits, loss = model(xb, yb) #forward pass
    optimizer.zero_grad(set_to_none=True) #zerar gradientes
    loss.backward() #backward pass
    optimizer.step() #atualizar parâmetros

#gerar a partir do modelo
context = torch.zeros((1, 1), dtype=torch.long, device=device) #contexto inicial
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist())) #gerar e decodificar texto
