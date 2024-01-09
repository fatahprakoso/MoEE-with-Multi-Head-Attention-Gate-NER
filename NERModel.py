from torch import nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Forward forget gate
        self.W_ffx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_ffx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_ffh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ffh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Forward cell gate
        self.W_cfx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_cfx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_cfh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_cfh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Forward input gate
        self.W_ifx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_ifx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_ifh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ifh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Forward output gate
        self.W_ofx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_ofx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_ofh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ofh = nn.Parameter(torch.Tensor(hidden_size))

        # Backward forget gate
        self.W_fbx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_fbx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_fbh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_fbh = nn.Parameter(torch.Tensor(hidden_size))
        
        # backward cell gate
        self.W_cbx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_cbx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_cbh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_cbh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Backward input gate
        self.W_ibx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_ibx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_ibh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ibh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Backward output gate
        self.W_obx = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.b_obx = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_obh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_obh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Set init weight to normal
        for name,p in self.named_parameters():
            nn.init.normal_(p)

    def forward(self, x):
        ''' BiLSTM forward
        Args:
            x (list of tensor): vectorize sentence that has been padded
        Returns:
            
        '''

        # Initialize hidden states and cell states for forward and backward LSTMs
        h_fx, c_fx = torch.zeros(self.hidden_size, device=device), torch.zeros(self.hidden_size, device=device)
        h_bx, c_bx = torch.zeros(self.hidden_size, device=device), torch.zeros(self.hidden_size, device=device)

        # Lists to store outputs
        outputs_fx, outputs_bx = [], []

        # Forward pass
        for i, x_i in enumerate(x):
            o_seq = []
            p_f = []
            p2_f = []
            p_ft = []
            p_it = []
            p_ot = []
            p_gt = []
            
            for t in range(x_i.size(0)):
                x_t = x[i][t]
                
                if all(x_t.isnan()):
                    o_seq.append(torch.zeros(self.hidden_size).to(device))
                    continue

                # Forward LSTM
                f_t = torch.sigmoid(x_t @ self.W_ffx.t() + self.b_ffx + h_fx @ self.W_ffh.t() + self.b_ffh)
                i_t = torch.sigmoid(x_t @ self.W_ifx.t() + self.b_ifx + h_fx @ self.W_ifh.t() + self.b_ifh)
                o_t = torch.sigmoid(x_t @ self.W_ofx.t() + self.b_ofx + h_fx @ self.W_ofh.t() + self.b_ofh)
                g_t = torch.tanh(x_t @ self.W_cfx.t() + self.b_cfx + h_fx @ self.W_cfh.t() + self.b_cfh)

                c_fx = f_t * c_fx + i_t * g_t
                h_fx = o_t * torch.tanh(c_fx)
    
                p_f.append(c_fx)
                p2_f.append(h_fx)
                p_ft.append(f_t)
                p_it.append(i_t)
                p_ot.append(o_t)
                p_gt.append(g_t)
                o_seq.append(h_fx)
            
            outputs_fx.append(torch.stack(o_seq))
            
        # Backward pass
        for i, x_i in enumerate(x):
            o_seq = []
            p_b = []
            p2_b = []
            pb_ft = []
            pb_it = []
            pb_ot = []
            pb_gt = []
            
            for t in range(x_i.size(0) - 1, -1, -1):
                x_t = x[i][t]
                
                if all(x_t.isnan()):
                    o_seq.append(torch.zeros(self.hidden_size).to(device))
                    continue

                # Backward LSTM
                f_t = torch.sigmoid(x_t @ self.W_fbx.t() + self.b_fbx + h_bx @ self.W_fbh.t() + self.b_fbh)
                i_t = torch.sigmoid(x_t @ self.W_ibx.t() + self.b_ibx + h_bx @ self.W_ibh.t() + self.b_ibh)
                o_t = torch.sigmoid(x_t @ self.W_obx.t() + self.b_obx + h_bx @ self.W_obh.t() + self.b_obh)
                g_t = torch.tanh(x_t @ self.W_cbx.t() + self.b_cbx + h_bx @ self.W_cbh.t() + self.b_cbh)

                c_bx = f_t * c_bx + i_t * g_t
                h_bx = o_t * torch.tanh(c_bx)
                
                p2_b.append(h_bx)
                p_b.append(c_bx)
                pb_ft.append(f_t)
                pb_it.append(i_t)
                pb_ot.append(o_t)
                pb_gt.append(g_t)
                o_seq.append(h_bx)
                    
            p_b.reverse()
            p2_b.reverse()
            pb_ft.reverse()
            pb_it.reverse()
            pb_ot.reverse()
            pb_gt.reverse()
            
            o_seq.reverse()
            outputs_bx.append(torch.stack(o_seq))
        
        outputs_fx = torch.stack(outputs_fx)
        outputs_bx = torch.stack(outputs_bx)
        
        # Concatenate hidden states from both directions
        outputs = torch.cat([outputs_fx, outputs_bx],2)
        
        return outputs
    
class AttentionGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Q = nn.Linear(hidden_size*2, input_size)
        self.K = nn.Linear(input_size, input_size)
        self.V = nn.Linear(input_size, input_size)
        
        for name, p in self.named_parameters():
            nn.init.normal_(p)
    
    def forward(self, q, k, v):
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        
        # matmul q k
        q_k = torch.bmm(q,k.transpose(1,2))/torch.sqrt(torch.Tensor([self.input_size]).to(device))
        q_k_softmax = F.softmax(q_k, dim=-1)
        
        # matmul o v
        o = torch.bmm(q_k_softmax, v)
        
        return o

class MultiheadAttentionGate(nn.Module):
    def __init__(self, input_size, hidden_size, n_head, ner_nums):
        """
            input size is output_expert_size
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        
        self.attentions = nn.ModuleList(AttentionGate(input_size*ner_nums, hidden_size) for _ in range(n_head))
        
        # output MoEE tidak harus sesuai dengan jumlah tag
        self.classifiers = nn.Linear(input_size*n_head*ner_nums, ner_nums)
        
        for name, p in self.named_parameters():
            nn.init.normal_(p)
        
    def forward(self, x, bi_lstm_o):
        o_a = []
        
        for i, att in enumerate(self.attentions):
            if i == 0: o = att(bi_lstm_o,x,x,verbose)
            else: o = att(bi_lstm_o,x,x,False)
            
            o_a.append(o)
            
        concatened_o = torch.cat(o_a,2)
        outputs = self.classifiers(concatened_o)
        
        return outputs
    
class MoEE(nn.Module):
    def __init__(self, ner_tags, hidden_size, expert_output_size):
        super().__init__()
        
        self.experts = nn.ModuleList([EntityExpert(hidden_size, expert_output_size) for _ in range(len(ner_tags))])
        
        # gate dilewatkan ke dropout terlebih dahulu
        
        self.gate = MultiheadAttentionGate(expert_output_size, hidden_size, 8, len(ner_tags))
    
    def forward(self, x):
        e_o = []
        
        for e in self.experts:
            e_output = e(x)
            e_o.append(e_output)
            
        outputs = torch.cat(e_o,2)   
        
        outputs = self.gate(outputs,x)
        
        return outputs

class EntityExpert(nn.Module):
    def __init__(self, hidden_state_size, output_size):
        super().__init__()
        
        self.entity_expert = nn.Linear(2*hidden_state_size, output_size)
        
        for name, p in self.named_parameters():
            nn.init.normal_(p)
            if 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, x):
        return self.entity_expert(x)

class CRF(nn.Module):
    def __init__(self, num_labels):
        super(CRF, self).__init__()
        self.num_labels = num_labels

        # Transisi dari label i ke label j (transisi[i, j] adalah transisi dari i ke j)
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))

    def forward(self, emissions, tags):
        batch_size, sentence_length, _ = emissions.size()

        # Compute the unary score
        unary_score = emissions.gather(2, tags.unsqueeze(2)).squeeze(2).sum(dim=1)

        # Compute the transition score
        transition_score = torch.zeros(batch_size)
        for i in range(sentence_length - 1):
            transition_score += self.transitions[tags[:, i], tags[:, i + 1]]

        # Sum of unary and transition scores
        total_score = unary_score + transition_score

        # Compute the partition function (Z)
        alpha = self.compute_alpha(emissions)
        log_partition = alpha[:, -1, :].logsumexp(dim=1).sum()

        # Compute the log likelihood
        loss = log_partition - total_score.sum()

        return loss

    def compute_alpha(self, emissions):
        batch_size, sentence_length, _ = emissions.size()
        alpha = torch.zeros(batch_size, sentence_length, self.num_labels)

        # Initialization
        alpha[:, 0, :] = emissions[:, 0, :]

        # Recursion
        for t in range(1, sentence_length):
            alpha[:, t, :] = emissions[:, t, :] + torch.logsumexp(alpha[:, t - 1, :] + self.transitions, dim=1)

        return alpha

    def viterbi_decode(self, emissions):
        batch_size, sentence_length, _ = emissions.size()

        # Initialize Viterbi variables
        delta = torch.zeros(batch_size, sentence_length, self.num_labels)
        backpointer = torch.zeros(batch_size, sentence_length, self.num_labels, dtype=torch.long)

        # Initialization
        delta[:, 0, :] = emissions[:, 0, :]

        # Recursion
        for t in range(1, sentence_length):
            trans_score = delta[:, t - 1, :].unsqueeze(2) + self.transitions
            max_scores, backpointer[:, t, :] = trans_score.max(dim=1)
            delta[:, t, :] = emissions[:, t, :] + max_scores

        # Termination
        best_last_tag = delta[:, -1, :].argmax(dim=1)

        # Backtrack to find best path using backpointers
        best_path = torch.zeros(batch_size, sentence_length, dtype=torch.long)
        best_path[:, -1] = best_last_tag

        for t in range(sentence_length - 2, -1, -1):
            best_path[:, t] = backpointer[:, t + 1, best_path[:, t + 1]]

        return best_path
    
class NERMOEE(nn.Module):
    def __init__(self, embedding_size, hidden_size, ner_tags, expert_output_size, batch=32):
        super().__init__()
        
        """
        Args:
            embedding_size     : Size of a vectorized token
            hidden_size        : Bi-LSTM output size
            ner_tags           : List of unique tags
            expert_output_size : Expert layer output size
            batch              : Batch for training
        """
        
        torch.manual_seed(34)
        
        self.encoder = BiLSTM(embedding_size, hidden_size)
        self.MoEE = MoEE(ner_tags, hidden_size, expert_output_size)
        self.CRF = CRF(len(ner_tags))
    
    def forward(self, x, labels, verbose):
        """
        Args:
            x       : Training data
            labels  : Target label coresponden with traning data.
            verbose : A boolean parameter that controls whether the training process are printed or not.
        """
        
        o = self.encoder(x, verbose)
        o = self.MoEE(o, verbose)
        o = self.CRF(o, labels, verbose)
        
        return o
    
    def predict(self, x):
        o = self.encoder(x,False)
        o = self.MoEE(o,False)
        o = self.CRF.viterbi_decode(o)
        
        return o
    
    def train(self, x, labels, verbose):
        return self.forward(x, labels, verbose)
    
    def verbose(self):
        self.encoder.get_params()