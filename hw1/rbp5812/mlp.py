import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()
    def _relu(self, x):
        return torch.relu(x)

    def _relu_deriv(self, x):
        return (x > 0).float()

    def _sigmoid(self, x):
        return torch.sigmoid(x)

    def _sigmoid_deriv(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def _identity(self, x):
        return x

    def _identity_deriv(self, x):
        return torch.ones_like(x)
    
    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        
        # TODO: Implement the forward function
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        z1 = x @ W1.T + b1  # x: (batch_size, in_features), W1: (out, in) -> W1.T: (in, out) -> z1: (batch_size, out)

                # Activation f
        if self.f_function == 'relu':
            a1 = self._relu(z1)
            f_deriv = self._relu_deriv
        elif self.f_function == 'sigmoid':
            a1 = self._sigmoid(z1)
            f_deriv = self._sigmoid_deriv
        elif self.f_function == 'identity':
            a1 = self._identity(z1)
            f_deriv = self._identity_deriv
        else:
            raise ValueError("Unsupported activation function f")
        
        # Linear 2
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        z2 = a1 @ W2.T + b2 # a1: (batch_size, linear_1_out_features), W2: (out, in) -> W2.T: (in, out) -> z2: (batch_size, linear_2_out_features)

        # Activation g
        if self.g_function == 'relu':
            y_hat = self._relu(z2)
            g_deriv = self._relu_deriv
        elif self.g_function == 'sigmoid':
            y_hat = self._sigmoid(z2)
            g_deriv = self._sigmoid_deriv
        elif self.g_function == 'identity':
            y_hat = self._identity(z2)
            g_deriv = self._identity_deriv
        else:
            raise ValueError("Unsupported activation function g")
        
        self.cache = {
            'x': x,
            'z1': z1, 'a1': a1, 'f_deriv': f_deriv,
            'z2': z2, 'y_hat': y_hat, 'g_deriv': g_deriv,
            'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2
        }

        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        # Retrieve values from cache
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        f_deriv = self.cache['f_deriv']
        z2 = self.cache['z2']
        y_hat = self.cache['y_hat']
        g_deriv = self.cache['g_deriv']
        W1 = self.cache['W1']
        W2 = self.cache['W2']
        b1 = self.cache['b1']
        b2 = self.cache['b2']

        batch_size = x.shape[0]

        # --- Gradient for the second linear layer and g ---

        # Gradient of loss with respect to z2: dJ/dz2 = dJ/dy_hat * dy_hat/dz2
        # where dy_hat/dz2 is the derivative of the g function evaluated at z2
        dJdz2 = dJdy_hat * g_deriv(z2) # Element-wise multiplication

        # Gradient of loss with respect to W2: dJ/dW2 = dJ/dz2 * dz2/dW2
        self.grads['dJdW2'] = dJdz2.T @ a1 

        # Gradient of loss with respect to b2: dJ/db2 = dJ/dz2 * dz2/db2
        # dz2/db2 = 1 (element-wise, for each bias term)
        # dJdb2 = sum(dJdz2, axis=0) across batch dimension
        self.grads['dJdb2'] = torch.sum(dJdz2, dim=0) # Sum over batch

        # Gradient of loss with respect to a1: dJ/da1 = dJ/dz2 * dz2/da1
        dJda1 = dJdz2 @ W2 

        # Gradient of loss with respect to z1: dJ/dz1 = dJ/da1 * da1/dz1
        # where da1/dz1 is the derivative of the f function evaluated at z1
        dJdz1 = dJda1 * f_deriv(z1) # Element-wise multiplication

        # Gradient of loss with respect to W1: dJ/dW1 = dJ/dz1 * dz1/dW1
        # dz1/dW1 = x.T (for a single sample, for the correct shape)
        # dJdW1 = x.T @ dJdz1
        self.grads['dJdW1'] = dJdz1.T @ x    # (out1, in1)

        # Gradient of loss with respect to b1: dJ/db1 = dJ/dz1 * dz1/db1
        # dz1/db1 = 1 (element-wise)
        # dJdb1 = sum(dJdz1, axis=0) across batch dimension
        self.grads['dJdb1'] = torch.sum(dJdz1, dim=0) # Sum over batch

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

        
def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    error = y_hat - y
    loss = torch.mean(error ** 2)  # scalar
    dJdy_hat = 2 * error / (y.shape[0] * y.shape[1])  # divide by total elements
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    eps = 1e-9
    y_hat = torch.clamp(y_hat, eps, 1. - eps)

    # loss
    loss = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))

    # gradient
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat) * y.numel())

    return loss, dJdy_hat












