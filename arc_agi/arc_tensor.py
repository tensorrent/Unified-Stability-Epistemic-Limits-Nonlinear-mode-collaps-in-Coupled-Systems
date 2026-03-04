"""
IntTensor & Integer Einstein Summation Engine
For deterministic, non-probabilistic multi-dimensional geometry.
"""

import itertools
from copy import deepcopy

class IntTensor:
    """A lightweight wrapper for multi-dimensional nested integer lists."""
    def __init__(self, data):
        self.data = deepcopy(data)
        self.shape = self._get_shape(self.data)

    def _get_shape(self, obj):
        if not isinstance(obj, list):
            return ()
        if len(obj) == 0:
            return (0,)
        return (len(obj),) + self._get_shape(obj[0])
    
    def to_list(self):
        return deepcopy(self.data)

def _get_element(tensor, indices):
    """Retrieve an element from a nested list given a tuple of indices."""
    curr = tensor
    for i in indices:
        curr = curr[i]
    return curr

def _set_element(tensor, indices, val):
    """Set an element in a nested list given a tuple of indices."""
    curr = tensor
    for i in indices[:-1]:
        curr = curr[i]
    curr[indices[-1]] = val

def _create_empty_tensor(shape):
    """Create a nested list of zeros with the given shape."""
    if not shape:
        return 0
    if len(shape) == 1:
        return [0] * shape[0]
    return [_create_empty_tensor(shape[1:]) for _ in range(shape[0])]

def int_einsum(subscripts: str, *operands) -> list:
    """
    Deterministic Integer Einstein Summation.
    Evaluates einsum notation strictly using integers to prevent precision loss.
    Example: 'ij,jk->ik' (Matrix Multiply)
             'ij->ji'    (Transpose)
             'ii->'      (Trace)
    """
    input_str, output_str = subscripts.replace(" ", "").split("->")
    input_terms = input_str.split(",")
    
    if len(input_terms) != len(operands):
        raise ValueError("Number of subscript terms must match over operands.")
        
    shapes = []
    dim_sizes = {}
    
    # Resolve dimensions
    for term, op in zip(input_terms, operands):
        if isinstance(op, IntTensor):
            shape = op.shape
            op_data = op.data
        else:
            op_tensor = IntTensor(op)
            shape = op_tensor.shape
            op_data = op_tensor.data
            
        shapes.append(shape)
        if len(term) != len(shape):
            raise ValueError(f"Term '{term}' expects {len(term)} dimensions, but operand has shape {shape}.")
            
        for char, dim in zip(term, shape):
            if char in dim_sizes:
                if dim_sizes[char] != dim:
                    raise ValueError(f"Dimension mismatch for index '{char}'.")
            else:
                dim_sizes[char] = dim
                
    out_shape = tuple(dim_sizes[c] for c in output_str)
    result = _create_empty_tensor(out_shape)
    
    # Evaluate all possible coordinate spaces
    all_indices = sorted(list(dim_sizes.keys()))
    ranges = [range(dim_sizes[idx]) for idx in all_indices]
    
    for coord in itertools.product(*ranges):
        coord_map = dict(zip(all_indices, coord))
        
        # Calculate product
        product = 1
        for term, op in zip(input_terms, operands):
            op_indices = tuple(coord_map[c] for c in term)
            # Fetch element dynamically
            if isinstance(op, IntTensor):
                 val = _get_element(op.data, op_indices)
            else:
                 val = _get_element(op, op_indices)
            product *= val
            
        # Accumulate
        if output_str:
            out_indices = tuple(coord_map[c] for c in output_str)
            curr = _get_element(result, out_indices)
            _set_element(result, out_indices, curr + product)
        else:
            result += product # Scalar accumulation

    return result

def einsum_affine(grid: list[list[int]], transform_matrix: list[list[int]]) -> list[list[int]]:
    """
    Apply a 2D affine transformation utilizing the integer tensor mechanics.
    Formula: out_val[i, j] = grid[ i*T[0,0] + j*T[0,1], i*T[1,0] + j*T[1,1] ]
    This provides spatial restructuring (rotations, reflections) directly via a tensor.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    if h == 0 or w == 0: return grid
    
    out = _create_empty_tensor((h, w))
    for i in range(h):
        for j in range(w):
            src_i = i * transform_matrix[0][0] + j * transform_matrix[0][1]
            src_j = i * transform_matrix[1][0] + j * transform_matrix[1][1]
            # Wrap around utilizing modulus for closed integer manifolds
            out[i][j] = grid[src_i % h][src_j % w]
    return out

def einsum_color_map(grid: list[list[int]], color_matrix: list[list[int]]) -> list[list[int]]:
    """
    Vectorized color transformation using standard Einsum evaluation.
    Converts each pixel value using a 10x10 mapping tensor.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    
    out = _create_empty_tensor((h, w))
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            # Matrix dot product essentially pulling the new value
            out[r][c] = color_matrix[val][val] # Diagonal map or specific projection
    return out
