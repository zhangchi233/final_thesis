from dcn.modules.deform_conv import *
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
class DeformConvPack_d_lora(DeformConvPack_d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        DeformConvPack_d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # print("in init")
        # embed()
        # Actual trainable parameters
        if type(kernel_size) is int:
            h,w,d = kernel_size, kernel_size, kernel_size  
        else:
            d,h,w = kernel_size
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_channels*w*d))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*h, r))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        DeformConvPack_d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True): # True for train and False for eval
 
        DeformConvPack_d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            # print("test")
            # embed()
            if self.merge_weights and not self.merged:
                # print("merging")
                # embed()
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):

        if self.r > 0 and not self.merged:

           
             
            self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            
        
        return DeformConvPack_d.forward(self, x)