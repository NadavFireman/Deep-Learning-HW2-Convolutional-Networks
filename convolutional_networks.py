"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import torch.nn.functional as F
from a3_helper import softmax_loss


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code

        stride = conv_param['stride']
        pad = conv_param['pad']
        N, C, H, W = x.shape
        F_filters, _, HH, WW = w.shape
        
        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W + 2 * pad - WW) // stride
        
        x_padded = F.pad(x, (pad, pad, pad, pad))
        
        out = torch.zeros(N, F_filters, H_out, W_out, dtype=x.dtype, device=x.device)
        
        for n in range(N):
            for f in range(F_filters):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                        out[n, f, i, j] = (window * w[f]).sum() + b[f]


        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        
        x, w, b, conv_param = cache
        stride = conv_param['stride']
        pad = conv_param['pad']

        N, C, H, W = x.shape
        F_filters, _, HH, WW = w.shape
        _, _, H_out, W_out = dout.shape

        x_padded = F.pad(x, (pad, pad, pad, pad))
        dx_padded = torch.zeros_like(x_padded)

        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)

        for n in range(N):
            for f in range(F_filters):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                        db[f] += dout[n, f, i, j]
                        dw[f] += dout[n, f, i, j] * window
                        dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += dout[n, f, i, j] * w[f]

        dx = dx_padded[:, :, pad:pad+H, pad:pad+W]
        
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']

        N, C, H, W = x.shape
        H_out = 1 + (H - pool_height) // stride
        W_out = 1 + (W - pool_width) // stride

        out = torch.zeros(N, C, H_out, W_out, dtype=x.dtype, device=x.device)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width]
                        out[n, c, i, j] = window.max()
        
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        
        x, pool_param = cache
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']

        N, C, H, W = x.shape
        _, _, H_out, W_out = dout.shape

        dx = torch.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width]
                        # Find position of the max in this window
                        max_val = window.max()
                        mask = (window == max_val)
                        # Pass the gradient only to the max position
                        dx[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width] += mask * dout[n, c, i, j]
        
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx
