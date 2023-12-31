import torch
import torch.nn as nn
        
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size,device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [],[]   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
            self.C.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C

class decoder_D(nn.Module):
    #def __init__(self, nc=1, nf=32):
    def __init__(self, nc=3, nf=64):
        super(decoder_D, self).__init__()
        self.d1 = nn.Conv2d(in_channels=nf, out_channels=nc, kernel_size=(1,1), stride=1, padding=0)  #(nc) x 64 x 64

    def forward(self, input):      
        d1 = self.d1(input)
        
        return d1  
    
class EncoderRNN(torch.nn.Module):
    def __init__(self,convcell, device):
        super(EncoderRNN, self).__init__()     
        self.decoder_D = decoder_D()  # general decoder 64x64x64 -> 64x64x3    
        self.decoder_D = self.decoder_D.to(device)
        self.convcell = convcell.to(device)

    def forward(self, input, first_timestep=False):  

        hidden1, output1 = self.convcell(input, first_timestep)  
        output_image = torch.sigmoid(self.decoder_D(output1[-1]))
        return output_image        