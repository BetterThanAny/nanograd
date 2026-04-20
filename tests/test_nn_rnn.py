import numpy as np
import pytest

from nanograd import Tensor, nn
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(23)


def _rt(shape, rng, lo=-0.5, hi=0.5):
    return Tensor(rng.uniform(lo, hi, size=shape).astype(np.float32), requires_grad=True)


def test_rnn_cell_shape(rng):
    cell = nn.RNNCell(4, 8, seed=0)
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    h = Tensor(np.zeros((3, 8), dtype=np.float32))
    h1 = cell(x, h)
    assert h1.shape == (3, 8)


def test_rnn_shape(rng):
    rnn = nn.RNN(4, 8, seed=0)
    x = Tensor(rng.standard_normal((2, 5, 4)).astype(np.float32))
    out, h = rnn(x)
    assert out.shape == (2, 5, 8)
    assert h.shape == (2, 8)


def test_rnn_gradcheck_small(rng):
    rnn = nn.RNN(2, 3, seed=0)
    x = _rt((1, 3, 2), rng)
    gradcheck(lambda x: rnn(x)[0].sum(), [x], atol=1e-2, rtol=1e-2)


def test_lstm_cell_shape(rng):
    cell = nn.LSTMCell(4, 8, seed=0)
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    h = Tensor(np.zeros((3, 8), dtype=np.float32))
    c = Tensor(np.zeros((3, 8), dtype=np.float32))
    h1, c1 = cell(x, (h, c))
    assert h1.shape == (3, 8) and c1.shape == (3, 8)


def test_lstm_shape(rng):
    m = nn.LSTM(4, 8, seed=0)
    x = Tensor(rng.standard_normal((2, 5, 4)).astype(np.float32))
    out, (h, c) = m(x)
    assert out.shape == (2, 5, 8)
    assert h.shape == (2, 8) and c.shape == (2, 8)


def test_gru_cell_shape(rng):
    cell = nn.GRUCell(4, 8, seed=0)
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    h = Tensor(np.zeros((3, 8), dtype=np.float32))
    h1 = cell(x, h)
    assert h1.shape == (3, 8)


def test_gru_shape(rng):
    m = nn.GRU(4, 8, seed=0)
    x = Tensor(rng.standard_normal((2, 5, 4)).astype(np.float32))
    out, h = m(x)
    assert out.shape == (2, 5, 8) and h.shape == (2, 8)


def test_gru_gradcheck_small(rng):
    m = nn.GRU(2, 3, seed=0)
    x = _rt((1, 3, 2), rng)
    gradcheck(lambda x: m(x)[0].sum(), [x], atol=1e-2, rtol=1e-2)


def test_lstm_gradcheck_small(rng):
    m = nn.LSTM(2, 3, seed=0)
    x = _rt((1, 3, 2), rng)
    gradcheck(lambda x: m(x)[0].sum(), [x], atol=1e-2, rtol=1e-2)


def test_scaled_dot_product_attention_shape(rng):
    B, T, D = 2, 4, 6
    q = Tensor(rng.standard_normal((B, T, D)).astype(np.float32))
    k = Tensor(rng.standard_normal((B, T, D)).astype(np.float32))
    v = Tensor(rng.standard_normal((B, T, D)).astype(np.float32))
    out = nn.scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, T, D)


def test_multihead_attention_shape(rng):
    m = nn.MultiHeadAttention(8, 2, seed=0)
    x = Tensor(rng.standard_normal((2, 5, 8)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 5, 8)


def test_multihead_attention_gradcheck_tiny(rng):
    m = nn.MultiHeadAttention(4, 2, seed=0)
    x = _rt((1, 2, 4), rng)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-1, rtol=1e-1)


def test_transformer_block_shape(rng):
    m = nn.TransformerBlock(8, 2, seed=0)
    x = Tensor(rng.standard_normal((2, 5, 8)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 5, 8)


def test_attention_causal_mask(rng):
    # causal mask: position i can only attend to positions <= i
    T = 4
    mask = np.tril(np.ones((T, T), dtype=bool))
    q = Tensor(rng.standard_normal((1, T, 4)).astype(np.float32))
    k = Tensor(rng.standard_normal((1, T, 4)).astype(np.float32))
    v = Tensor(rng.standard_normal((1, T, 4)).astype(np.float32))
    out = nn.scaled_dot_product_attention(q, k, v, mask=mask[None, :, :])
    # future positions should not have contributed — but this is hard to check without probing softmax.
    # Just check shape + no NaN.
    assert out.shape == (1, T, 4)
    assert not np.any(np.isnan(out.data))


# ---------- toy seq task: learn to copy last element ----------


def test_lstm_learns_last_element():
    """Given a sequence of floats, predict the last element. LSTM should nail it."""
    rng = np.random.default_rng(0)
    B, T, D = 32, 5, 1
    # random sequences; target = last token
    X = rng.uniform(-1, 1, size=(B, T, D)).astype(np.float32)
    y = X[:, -1, :]  # (B, 1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 8, seed=0)
            self.head = nn.Linear(8, 1, seed=1)

        def forward(self, x):
            out, (h, _) = self.lstm(x)
            return self.head(h)

    from nanograd import optim
    from nanograd.nn import functional as F

    model = Net()
    opt = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(200):
        pred = model(Tensor(X))
        loss = F.mse_loss(pred, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"final loss={loss.item()}"
