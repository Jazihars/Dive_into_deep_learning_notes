# PyTorch自动求导的一些知识

## 使用PyTorch的backward()函数来求导数
在空白脚本中测试如下的代码：
``` python
import torch

x = torch.arange(4.0, requires_grad=True)

y = 2 * torch.dot(x, x)

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)

y.backward()

print("----------------------求梯度后----------------------")

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("x.grad == 4 * x: ", x.grad == 4 * x)
```
结果为：
```
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  None
----------------------求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  tensor([ 0.,  4.,  8., 12.])
x.grad == 4 * x:  tensor([True, True, True, True])
```
因为
$$
y=2x^2
$$
所以
$$
\frac{\mathrm{d} y}{\mathrm{d} x}=4x
$$
所以上述代码中`x.grad == 4 * x`表达式的值为`tensor([True, True, True, True])`

## 梯度清零
在空白脚本中测试如下的代码：
``` python
import torch

x = torch.arange(4.0, requires_grad=True)

y = 2 * torch.dot(x, x)

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)

y.backward()

print("----------------------求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("x.grad == 4 * x: ", x.grad == 4 * x)


x.grad.zero_()
y = x.sum()
y.backward()
print("----------------------第二次求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
```
结果为：
```
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  None
----------------------求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  tensor([ 0.,  4.,  8., 12.])
x.grad == 4 * x:  tensor([True, True, True, True])
----------------------第二次求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(6., grad_fn=<SumBackward0>)
x.grad:  tensor([1., 1., 1., 1.])
```
我们看到，执行梯度清零`x.grad.zero_()`后，得出了正确的结果。
我们再来测试一下，如果不执行梯度清零，会发生什么。把`x.grad.zero_()`注释掉之后，在空白脚本中测试如下的代码：
``` python
import torch

x = torch.arange(4.0, requires_grad=True)

y = 2 * torch.dot(x, x)

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)

y.backward()

print("----------------------求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("x.grad == 4 * x: ", x.grad == 4 * x)


# x.grad.zero_()
y = x.sum()
y.backward()
print("----------------------第二次求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
```
结果为：
```
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  None
----------------------求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  tensor([ 0.,  4.,  8., 12.])
x.grad == 4 * x:  tensor([True, True, True, True])
----------------------第二次求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(6., grad_fn=<SumBackward0>)
x.grad:  tensor([ 1.,  5.,  9., 13.])
```
可以看到，两次求导的结果被累加起来了。所以，一般来说，我们需要执行梯度清零操作。

## 我们一般都是对标量求梯度
在空白脚本中测试如下的代码：
``` python
import torch

x = torch.arange(4.0, requires_grad=True)

y = 2 * torch.dot(x, x)

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)

y.backward()

print("----------------------求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("x.grad == 4 * x: ", x.grad == 4 * x)


x.grad.zero_()
y = x * x
y.sum().backward()
print("----------------------第二次求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("x.grad == 2 * x: ", x.grad == 2 * x)
```
结果为：
```
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  None
----------------------求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  tensor([ 0.,  4.,  8., 12.])
x.grad == 4 * x:  tensor([True, True, True, True])
----------------------第二次求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)
x.grad:  tensor([0., 2., 4., 6.])
x.grad == 2 * x:  tensor([True, True, True, True])
```
可以看到，我们通过`y.sum().backward()`，先把`y`变成了一个标量，然后再对`y`求梯度。我们来更详细地解释一下。 因为`x`是一个四维的张量，因此，可以把`x`记作：
$$
x=\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{pmatrix}
$$
因为`y = x * x`，所以
$$
y=x^2=\begin{pmatrix}
x_1^2 \\
x_2^2 \\
x_3^2 \\
x_4^2
\end{pmatrix}
$$
所以，对`y`求导数，结果为：
$$
\frac{\mathrm{d} y}{\mathrm{d} x}=\begin{pmatrix}
2x_1 \\
2x_2 \\
2x_3 \\
2x_4
\end{pmatrix}=2x
$$
这就是最后打印出来的结果`x.grad == 2 * x:  tensor([True, True, True, True])`的详细解释。

我们来试试，假如直接对`y`求梯度，会发生什么。在空白脚本中测试如下的代码：
``` python
import torch

x = torch.arange(4.0, requires_grad=True)

y = 2 * torch.dot(x, x)

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)

y.backward()

print("----------------------求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("x.grad == 4 * x: ", x.grad == 4 * x)


x.grad.zero_()
y = x * x
y.backward()
print("----------------------第二次求梯度后----------------------")
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
```
结果为：
```
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  None
----------------------求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor(28., grad_fn=<MulBackward0>)
x.grad:  tensor([ 0.,  4.,  8., 12.])
x.grad == 4 * x:  tensor([True, True, True, True])
Traceback (most recent call last):
  File "test.py", line 22, in <module>
    y.backward()
  File "/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/tensor.py", line 245, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/autograd/__init__.py", line 141, in backward
    grad_tensors_ = _make_grads(tensors, grad_tensors_)
  File "/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/autograd/__init__.py", line 50, in _make_grads
    raise RuntimeError("grad can be implicitly created only for scalar outputs")
RuntimeError: grad can be implicitly created only for scalar outputs
```
我们看到，PyTorch报错了。PyTorch只能对标量求导。

## 把某些参数作为常数固定住
在空白脚本里测试如下的代码：
``` python
import torch

x = torch.arange(4.0, requires_grad=True)
y = x * x
u = y.detach()  # 把y作为一个常数，而不是一个关于x的函数，赋值给u。因此，u的值就是x*x这个张量（u是一个常数张量）。
z = u * x

z.sum().backward()

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("u: ", u)
print("x.grad == u: ", x.grad == u)
```
结果为：
```
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)
x.grad:  tensor([0., 1., 4., 9.])
u:  tensor([0., 1., 4., 9.])
x.grad == u:  tensor([True, True, True, True])
```
可以看到，`u = y.detach()`这行代码是把`y`作为一个常数，而不是一个关于`x`的函数，赋值给`u`。因此，`u`的值就是`x*x`这个常数张量`tensor([0., 1., 4., 9.])`。对这一点，我们再来详细地解释一下。因为
$$
x=\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{pmatrix}=
\begin{pmatrix}
0 \\
1 \\
2 \\
3
\end{pmatrix}
$$
所以，`y = x * x`和`u = y.detach()`这两行代码的执行效果就是：
$$
y=\begin{pmatrix}
x_1^2 \\
x_2^2 \\
x_3^2 \\
x_4^2
\end{pmatrix},
u=\begin{pmatrix}
0 \\
1 \\
4 \\
9
\end{pmatrix}
$$
因为`z = u * x`，`u`是一个常数向量。所以
$$
z=\begin{pmatrix}
0 \\
x_2 \\
4x_3 \\
9x_4
\end{pmatrix}
$$


`z.sum()`的结果就是：
$$
z.sum()=x_2+4x_3+9x_4
$$
因此，`z.sum().backward()`对`x`求导数之后，结果为：
$$
x.grad=\begin{pmatrix}
0 \\
1 \\
4 \\
9
\end{pmatrix}
$$
这就是最后结果`x.grad == u:  tensor([True, True, True, True])`的详细解释。

我们再在空白脚本中测试一下下述代码：
``` python
import torch

x = torch.arange(4.0, requires_grad=True)
y = x * x
u = y.detach()  # 把y作为一个常数，而不是一个关于x的函数，赋值给u。因此，u的值就是x*x这个张量（u是一个常数张量）。
z = u * x

z.sum().backward()

print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("u: ", u)
print("x.grad == u: ", x.grad == u)

print("----------------------第一次求梯度后----------------------")

x.grad.zero_()
y.sum().backward()
print("x: ", x)
print("y: ", y)
print("x.grad: ", x.grad)
print("x.grad == 2 * x: ", x.grad == 2 * x)
```
结果为：
```
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)
x.grad:  tensor([0., 1., 4., 9.])
u:  tensor([0., 1., 4., 9.])
x.grad == u:  tensor([True, True, True, True])
----------------------第一次求梯度后----------------------
x:  tensor([0., 1., 2., 3.], requires_grad=True)
y:  tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)
x.grad:  tensor([0., 2., 4., 6.])
x.grad == 2 * x:  tensor([True, True, True, True])
```
把`x`的梯度清零以后，由于`y = x * x`，所以还可以继续让`y`对`x`求导数。因为
$$
y=x^2=\begin{pmatrix}
x_1^2 \\
x_2^2 \\
x_3^2 \\
x_4^2
\end{pmatrix}
$$
所以
$$
y.sum()=x_1^2+x_2^2+x_3^2+x_4^2
$$
所以`y.sum()`对`x`求导后，结果为：
$$
x.grad=\begin{pmatrix}
2x_1 \\
2x_2 \\
2x_3 \\
2x_4
\end{pmatrix}=2x=
\begin{pmatrix}
0 \\
2 \\
4 \\
6
\end{pmatrix}
$$
这就是上述代码最后输出的结果：
```
x.grad:  tensor([0., 2., 4., 6.])
x.grad == 2 * x:  tensor([True, True, True, True])
```
的详细解释。

## 利用PyTorch对一个复杂的函数控制流来求导
在空白脚本中测试如下的代码：
``` python
import torch


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)  # size=()代表a是一个标量
d = f(a)
d.backward()

print("a: ", a)
print("d: ", d)
print("a.grad: ", a.grad)
print("a.grad == d / a: ", a.grad == d / a)
```
结果为：
```
a:  tensor(1.1633, requires_grad=True)
d:  tensor(1191.2576, grad_fn=<MulBackward0>)
a.grad:  tensor(1024.)
a.grad == d / a:  tensor(True)
```
我们来详细解释一下为什么有`a.grad == d / a:  tensor(True)`。因为`d = f(a)`，所以要么是`d = 100 * b`，要么是`d = b`。因为
$$
b=\underbrace{2 \times 2 \times \cdots \times 2}_{\geq{0\text{个}2}} \times 2a
$$
所以
$$
d=100 \times \underbrace{2 \times 2 \times \cdots \times 2}_{\geq{0\text{个}2}} \times 2a
$$
或者
$$
d=\underbrace{2 \times 2 \times \cdots \times 2}_{\geq{0\text{个}2}} \times 2a
$$
必有一个成立。所以
$$
\frac{\mathrm{d} d}{\mathrm{d} a}=\frac{d}{a}
$$
这就是对`a.grad == d / a:  tensor(True)`这句输出结果的详细解释。