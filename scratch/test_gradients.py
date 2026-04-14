import tensorflow as tf
from core.neuron import ConvLIFCortexLayer, LIFCortexLayer, DeconvLIFCortexLayer

def test_gradient_warnings():
    print("> Testing Differentiable Connectivity...")
    
    # Initialize layers
    lif = LIFCortexLayer(10, 5)
    conv = ConvLIFCortexLayer((16,16,1), 4, 3)
    deconv = DeconvLIFCortexLayer((8,8,1), 4, 3)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Fake input batch
    x_lif = tf.random.normal((1, 10, 10))
    x_conv = tf.random.normal((1, 10, 16*16*1))
    
    # We only test LIF and CONV since DECONV is purposefully bypassed in backprop
    layers = [lif, conv]
    
    with tf.GradientTape() as tape:
        # Run forward pass
        out_lif = lif.forward(x_lif)
        out_conv = conv.forward(x_conv)
        
        # Calculate dummy loss
        loss = tf.reduce_mean(tf.square(out_lif)) + tf.reduce_mean(tf.square(out_conv))
    
    # Get variables
    vars = []
    for l in layers:
        vars.extend(l.get_variables())
    
    # Print variables being tracked
    print(f"> Tracked Variables ({len(vars)}):")
    for v in vars:
        print(f"  - {v.name}")
        
    # Calculate gradients
    grads = tape.gradient(loss, vars)
    
    # Check for None gradients (which cause the warning)
    for v, g in zip(vars, grads):
        if g is None:
            print(f"!!! WARNING: No gradient for {v.name}")
        else:
            print(f"[OK] Gradient exists for {v.name}")

    if any(g is None for g in grads):
        print("FAIL: Gradient warnings expected.")
    else:
        print("SUCCESS: No gradient warnings!")

if __name__ == "__main__":
    test_gradient_warnings()
