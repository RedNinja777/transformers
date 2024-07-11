


##########################################################
# 
# To use torch.optim you have to construct an optimizer object, that will hold the current state of the optimizer and will update the Parameters based on the computed gradients.
# state of an optimizer object includes:
# Note Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs. 
# Examples of objects that don’t satisfy those properties are sets and iterators over values of dictionaries.

# To construct an Optimizer you have to give it an iterable containing the parameters to optimize. 
# Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.

# If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it. 
# Parameters of a model after .cuda() will be different objects with those before the call.

# Optimizer s also support specifying per-parameter options. To do this, instead of passing an iterable of Parameters, pass in an iterable of dict. 
# Each of them will define a separate parameter group, and should contain a params key, containing a list of parameters belonging to it. 
# Other keys should match the keyword arguments accepted by the optimizers, and will be used as optimization options for this group.
# Note: You can still pass options as keyword arguments. They will be used as defaults, in the groups that didn’t override them. 

# All optimizers implement a step() method that updates the parameters. 

# weight_decay is a regularization technique somewhat similar to L2 regularization, in that it slightly reduces all weights gradually, pulling them closer towards zero.
# The purpose is to avoid overfitting by keeping wights and biases close to zero since it’s been discovered that overfitting often corresponds to large weights.
# L2 regularization and weight decay regularization are equivalent for standard SGD (when rescaled by the learning rate), 
# but this is not the case for adaptive gradient algorithms, such as Adam. (DECOUPLED WEIGHT DECAY REGULARIZATION)




##########################################################
# Scheduler
# torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.
# Learning rate scheduling should be applied (scheduler.step()) after optimizer’s update

# torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1) sets the learning rate of each parameter group to the initial lr times a given function. When last_epoch=-1, sets initial lr as lr.
# - lr_lambda (function or list) – A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.
# - last_epoch (int) – The index of last epoch. Default: -1.







##########################################################
# Mixed precision training
# An instance scaler of GradScaler is only useful when doing mixed precision, and helps perform the steps of gradient scaling conveniently.
#   - scaler.scale(loss) multiplies a given loss (in FP16) by scaler’s current scale factor (scaler.get_scale()) and return the scaled loss tensor, 
#     which can then call backward() to compute gradients (because gradients are also scaled up so they don't underlow to 0 in FP16).
#   - scaler.step(optimizer) safely unscales gradients and calls/or skips (due to inf or NaN) optimizer.step().
#     -- unscales means divide by the scale factor, in FP32 precision
#   - scaler.update() updates scaler’s scale factor for next iteration. 
#     -- Why needed? If optimizer.step() is skipped, this update() reduces the scale factor; otherwise leave the scale factor unchanged.
# Scale factor (typically 512) is used to make the loss numbers bigger so in FP16 they won't underflow to 0.

# In neural nets, all the computations are usually done in single precision (32-bit floating number), 
# which means all the floats in all the arrays that represent inputs, activations, weights… are 32-bit floats (FP32). 
# An idea to reduce memory usage has been to try and do the same thing in half-precision, which means using 16-bits floats (FP16). 
# By definition, they take half the space in RAM, and in theory could allow you to double the size of your model and double your batch size.
# The bfloat16 (Brain Floating Point) floating-point format is a computer number format occupying 16 bits in computer memory; 
# it represents a wide dynamic range of numeric values by using a floating radix point. 
# This format is a truncated (16-bit) version of the 32-bit IEEE 754 single-precision floating-point format (binary32) with the intent of accelerating machine learning and near-sensor computing. 
# It preserves the approximate dynamic range of 32-bit floating-point numbers by retaining 8 exponent bits, but supports only an 8-bit precision rather than the 24-bit significand of the binary32 format. 
# More so than single-precision 32-bit floating-point numbers, bfloat16 numbers are unsuitable for integer calculations, but this is not their intended use. 
# Bfloat16 is used to reduce the storage requirements and increase the calculation speed of machine learning algorithms.

# Another very nice feature is that NVIDIA developed its newer generation GPUs (since Volta) to take fully advantage of half-precision tensors. 
# Basically, if you give half-precision tensors to those, they’ll stack them so that each core can do more operations at the same time, and theoretically gives an 8x speed-up (sadly, just in theory).

# So training at half precision is better for your memory usage, way faster if you have a Volta GPU (still a tiny bit faster if you don’t have Volta since the computations are easiest). 
# How do we do it? Super easily in pytorch, we just have to put .half() everywhere: on the inputs of our model and all the parameters. 
# Problem is that you usually won’t see the same accuracy in the end (so it happens sometimes) because half-precision is… well… not as precise.

# Problems with half-precision:
# To understand the problems with half precision, let’s look briefly at what an FP16 looks like.            
# In FP16, The sign bit gives us +1 or -1, then we have 5 bits to code an exponent between -14 and 15, while the fraction part has the remaining 10 bits. 
# Compared to FP32, we have a smaller range of possible values (2e-14 to 2e15 roughly, compared to 2e-126 to 2e127 for FP32) but also a smaller offset.

# For instance, between 1 and 2, the FP16 format only represents the number 1, 1+2e-10, 1+2*2e-10… which means that 1 + 0.0001 = 1 in half precision. 
# That’s what will cause a certain numbers of problems, specifically three that can occur and mess up your training.
# 	1. The weight update is imprecise, because gradients of weights are typically several orders of magnitude smaller than weights and LR is small: 
#      inside your optimizer, you basically do w = w - lr * w.grad for each weight of your network. 
#      The problem in performing this operation in half precision is that very often, w.grad is several orders of magnitude below w, and the learning rate is also small. 
#      The situation where w=1 and lr*w.grad is 0.0001 (or lower) is therefore very common, but the update doesn’t do anything in those cases.
# 	2. The gradients can underflow. In FP16, your gradients can easily be replaced by 0 because they are too low.
# 	3. The activations or loss can overflow. The opposite problem from the gradients: it’s easier to hit nan (or infinity) in FP16 precision, and your training might more easily diverge. 
#      (Why? One reason may be caused by normalization when dividing std which might be a very small number. Another reason about loss?)

# The solution: mixed precision training
# To address those three problems, we don’t fully train in FP16 precision. As the name mixed training implies, some of the operations will be done in FP16, others in FP32. 
# This is mainly to take care of the first problem listed above. For the next two there are additional tricks.

# The main idea is that we want to do the forward pass and the gradient computation in half precision (to go fast) but the weight update in single precision (to be more precise). 
# It’s okay if w and grad are both half floats, but when we do the operation w = w - lr * grad, we need to compute it in FP32. 
# That way our 1 + 0.0001 is going to be 1.0001.

# This is why we keep a copy of the weights in FP32 (called master model). Then, our training loop will look like:
# 	1. compute the forward output with the FP16 model, then the loss
# 	2. back-propagate the gradients in half-precision.
# 	3. copy the gradients in FP32 precision
# 	4. do the update on the master model (in FP32 precision)
# 	5. copy the master model in the FP16 model.

# Note that we lose precision during step 5, and that the 1.0001 in one of the weights will go back to 1. 
# But if the next update corresponds to add 0.0001 again, since the optimizer step() is done on the master model, the 1.0001 will become 1.0002 
# and if we eventually go like this up to 1.0005, the FP16 model will be able to tell the difference.

# That takes care of problem 1. For the second problem, we use something called gradient scaling: 
# to avoid the gradients getting zeroed by the FP16 precision, we multiply the loss by a scale factor (scale=512 is a good value in our experiments). 
# That way we can make the gradients (after scaled up) bigger, and have them not become zero.

# Of course we don’t want those 512-scaled gradients to be in the weight update, so after converting gradients into FP32, 
# we can divide them by this scale factor (once they have no risks of becoming 0). This changes the loop above to:
# 	1. compute the output with the FP16 model, then the loss.
# 	2. multiply the loss by scale then back-propagate the gradients in half-precision. Note gradients are scaled the same way as loss.
# 	3. copy the gradients in FP32 precision then divide them by scale. This is called `unscale`` the gradients in FP32.
# 	4. do the update on the master model (in FP32 precision).
# 	5. copy the master model in the FP16 model.

# For the last problem, the tricks offered by NVIDIA are to leave the batchnorm layers in single precision (they don’t have many weights so it’s not a big memory challenge) 
# and compute the loss in single precision (which means converting the last output of the model such as softmax in single precision before passing it to the loss).

# By switching to 16-bit, we’ll be using half the memory and theoretically less computation at the expense of the available number range and precision. 
# However, pure 16-bit training creates a lot of problems for us (imprecise weight updates, gradient underflow and overflow). 
# Mixed precision training alleviate these problems.

