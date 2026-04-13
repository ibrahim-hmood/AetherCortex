import sys
import os

filepath = 'core/neuron.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace get_variables for LIFCortexLayer
old_lif = """    def get_variables(self):
        return [self.weights, self.biases]"""
new_lif = """    def get_variables(self):
        return [self.weights, self.biases, self.permanence]"""

# Replace get_variables for ConvLIFCortexLayer
old_conv = """    def get_variables(self):
        return [self.weights, self.biases]"""
# Note: Conv occurs after LIF, so we need to be careful with string.replace if it's identical.
# I will use a more unique anchor.

content = content.replace('    def get_variables(self):\n        return [self.weights, self.biases]', 
                         '    def get_variables(self):\n        return [self.weights, self.biases, self.permanence]')

# Fix Deconv as well
content = content.replace('    def get_variables(self):\n        return [self.weights, self.biases]', 
                         '    def get_variables(self):\n        return [self.weights, self.biases, self.permanence]')

with open(filepath, 'w') as f:
    f.write(content)

print("Neuro-Patch applied successfully.")
