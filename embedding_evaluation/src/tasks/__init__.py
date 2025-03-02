"""
Evaluation tasks for instruction embeddings.
"""

from .instruction_synonym import InstructionSynonymTask
from .semantic_block import SemanticBlockTask
from .dead_code import DeadCodeTask
from .enhanced_instruction_synonym import EnhancedSynonymTask
from .enhanced_semantic_block import StructureAwareBlockTask
from .enhanced_dead_code import DataFlowAwareDeadCodeTask
from .function_boundary import FunctionBoundaryTask
from .vulnerability import VulnerabilityDetectionTask

# Register available tasks
TASKS = {
    # Basic tasks
    'synonym': InstructionSynonymTask,
    'block': SemanticBlockTask,
    'dead_code': DeadCodeTask,
    
    # Enhanced tasks
    'enhanced_synonym': EnhancedSynonymTask,
    'enhanced_block': StructureAwareBlockTask,
    'enhanced_dead_code': DataFlowAwareDeadCodeTask,
    
    # New tasks
    'function_boundary': FunctionBoundaryTask,
    'vulnerability': VulnerabilityDetectionTask
}

def get_task(name, **kwargs):
    """
    Get an evaluation task by name.
    
    Args:
        name: Name of the task
        **kwargs: Additional parameters for the task
        
    Returns:
        BaseTask: The evaluation task
    """
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Available tasks: {list(TASKS.keys())}")
    
    return TASKS[name](**kwargs)