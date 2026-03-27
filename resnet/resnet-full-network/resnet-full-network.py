import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        # Projection shortcut if dimensions change
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Conv -> ReLU -> Conv -> Add Skip -> ReLU
        """
        # First layer and activation
        out = relu(np.dot(x, self.W1))
        
        # Second layer (no ReLU yet)
        out = np.dot(out, self.W2)
        
        # Process the skip connection (shortcut)
        if self.W_proj is not None:
            shortcut = np.dot(x, self.W_proj)
        else:
            shortcut = x
            
        # Add skip connection and apply final ReLU
        out = relu(out + shortcut)
        
        return out

class ResNet18:
    """
    Simplified ResNet-18 architecture.
    
    Structure:
    - conv1: 3 -> 64 channels
    - layer1: 2 BasicBlocks, 64 channels
    - layer2: 2 BasicBlocks, 128 channels (first block downsamples)
    - layer3: 2 BasicBlocks, 256 channels (first block downsamples)
    - layer4: 2 BasicBlocks, 512 channels (first block downsamples)
    - fc: 512 -> num_classes
    """
    
    def __init__(self, num_classes: int = 10):
        self.conv1 = np.random.randn(3, 64) * 0.01
        
        # Build layers - Grouped into lists for easier forward iteration
        self.layer1 = [
            BasicBlock(64, 64), 
            BasicBlock(64, 64)
        ]
        self.layer2 = [
            BasicBlock(64, 128, downsample=True), 
            BasicBlock(128, 128)
        ]
        self.layer3 = [
            BasicBlock(128, 256, downsample=True), 
            BasicBlock(256, 256)
        ]
        self.layer4 = [
            BasicBlock(256, 512, downsample=True), 
            BasicBlock(512, 512)
        ]
        
        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet-18.
        """
        # Initial convolution and activation
        x = relu(np.dot(x, self.conv1))
        
        # Pass sequentially through all residual blocks
        for block in self.layer1: x = block.forward(x)
        for block in self.layer2: x = block.forward(x)
        for block in self.layer3: x = block.forward(x)
        for block in self.layer4: x = block.forward(x)
        
        # Final fully connected layer (classification head)
        logits = np.dot(x, self.fc)
        
        return logits