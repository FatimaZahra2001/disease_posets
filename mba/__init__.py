""" MBA structural-MRI experiment.

MRI-specific construction:
    FreeSurfer -> disease evidence -> 27-node probability tensor

Official MBA:
    P -> Q1, Q2, Q3 via the vendored PyMBA implementation

Evaluation:
    nested CV -> representation comparison -> interpretation
"""

__version__ = "0.1.0"