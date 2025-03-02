"""
Enhanced task for evaluating dead code detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import defaultdict

from .dead_code import DeadCodeTask

class DataFlowAwareDeadCodeTask(DeadCodeTask):
    """Dead code detection enhanced with data flow analysis."""
    
    def __init__(self, impact_threshold=0.1, data_flow_weight=0.4):
        """
        Initialize the data flow aware dead code detection task.
        
        Args:
            impact_threshold: Threshold for semantic impact (below which code is considered "dead")
            data_flow_weight: Weight for data flow analysis in combined score
        """
        super().__init__(impact_threshold=impact_threshold)
        self.name = "Data Flow Aware Dead Code Detection"
        self.description = "Detect instructions with no semantic impact using data flow analysis"
        self.data_flow_weight = data_flow_weight
    
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on dead code detection.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Dictionary with keys:
                       - 'code_blocks': List of instruction blocks
                       - 'target_indices': List of indices of instructions to evaluate
                       - 'labels': List of boolean values indicating if the instruction at 
                                  the corresponding index is dead code
            
        Returns:
            dict: Evaluation results
        """
        code_blocks = test_data['code_blocks']
        target_indices = test_data['target_indices']
        true_labels = test_data['labels']
        
        # Calculate combined impact scores
        embedding_impacts = []
        data_flow_impacts = []
        combined_impacts = []
        
        for block, idx in zip(code_blocks, target_indices):
            # Calculate semantic impact using embeddings
            block_with = list(block)  # Create a copy
            removed_instr = block_with.pop(idx)
            
            # Skip if block is now empty
            if not block_with:
                # Edge case: if removing the instruction makes the block empty,
                # it was likely not dead code
                emb_impact = 1.0
                data_flow_impact = 1.0
                combined_impact = 1.0
            else:
                # Get embeddings
                vec_original = embedding_model.transform([block])[0]
                vec_without = embedding_model.transform([block_with])[0]
                
                # Calculate cosine similarity
                norm_orig = np.linalg.norm(vec_original)
                norm_without = np.linalg.norm(vec_without)
                
                if norm_orig == 0 or norm_without == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(vec_original, vec_without) / (norm_orig * norm_without)
                
                # Impact is inverse of similarity
                emb_impact = 1.0 - similarity
                
                # Calculate data flow impact
                data_flow_impact = self._calculate_data_flow_impact(block, idx)
                
                # Combined impact
                combined_impact = (1 - self.data_flow_weight) * emb_impact + self.data_flow_weight * data_flow_impact
            
            embedding_impacts.append(emb_impact)
            data_flow_impacts.append(data_flow_impact)
            combined_impacts.append(combined_impact)
        
        # Predict dead code based on impact threshold
        predictions = [impact < self.impact_threshold for impact in combined_impacts]
        
        return {
            'embedding_impacts': embedding_impacts,
            'data_flow_impacts': data_flow_impacts,
            'combined_impacts': combined_impacts,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def _calculate_data_flow_impact(self, block, target_idx):
        """
        Calculate data flow impact of an instruction.
        
        Args:
            block: Instruction block
            target_idx: Index of the target instruction
            
        Returns:
            float: Data flow impact score
        """
        # Extract target instruction
        target_instr = block[target_idx]
        
        # Analyze data flow
        reads, writes = self._analyze_data_dependencies(block)
        
        # Determine if target instruction's outputs are used later
        if target_idx not in writes:
            # If the instruction doesn't write to any registers, it might still
            # have side effects (like memory writes)
            return 0.5
        
        # Get registers written by the target instruction
        written_regs = writes[target_idx]
        
        # Check if these registers are read by later instructions
        is_used = False
        for idx in range(target_idx + 1, len(block)):
            if any(reg in reads[idx] for reg in written_regs):
                is_used = True
                break
        
        # If the written registers are never read later, likely dead code
        if not is_used:
            return 0.1
        
        # Check if the registers are overwritten before being read
        is_overwritten = False
        for idx in range(target_idx + 1, len(block)):
            if any(reg in writes[idx] for reg in written_regs):
                # If we haven't found a read yet, and it's being overwritten,
                # likely dead code
                if not is_used:
                    is_overwritten = True
                    break
        
        if is_overwritten:
            return 0.2
        
        # If the instruction's outputs are used, it's likely not dead code
        return 0.8
    
    def _analyze_data_dependencies(self, block):
        """
        Analyze data dependencies in an instruction block.
        
        Args:
            block: Instruction block
            
        Returns:
            tuple: (reads, writes) dictionaries mapping instruction indices to registers
        """
        reads = defaultdict(set)
        writes = defaultdict(set)
        
        for idx, instr in enumerate(block):
            # Skip if not a string
            if not isinstance(instr, str):
                continue
            
            # Split instruction into opcode and operands
            parts = instr.split(None, 1)
            opcode = parts[0].lower() if parts else ""
            
            if len(parts) < 2:
                continue
            
            operands = parts[1].split(',')
            
            # Analyze based on x86-64 instruction patterns
            # This is a simplified analysis for common patterns
            
            # MOV, PUSH, POP instructions
            if opcode in ['mov', 'movzx', 'movsx']:
                # Format: mov dest, src
                if len(operands) >= 2:
                    dest = operands[0].strip()
                    src = operands[1].strip()
                    
                    # Extract registers
                    dest_regs = self._extract_registers(dest)
                    src_regs = self._extract_registers(src)
                    
                    # Add to reads and writes
                    reads[idx].update(src_regs)
                    writes[idx].update(dest_regs)
            
            elif opcode in ['push']:
                # Format: push src
                if operands:
                    src = operands[0].strip()
                    src_regs = self._extract_registers(src)
                    reads[idx].update(src_regs)
            
            elif opcode in ['pop']:
                # Format: pop dest
                if operands:
                    dest = operands[0].strip()
                    dest_regs = self._extract_registers(dest)
                    writes[idx].update(dest_regs)
            
            # Arithmetic instructions
            elif opcode in ['add', 'sub', 'and', 'or', 'xor']:
                # Format: add dest, src
                if len(operands) >= 2:
                    dest = operands[0].strip()
                    src = operands[1].strip()
                    
                    dest_regs = self._extract_registers(dest)
                    src_regs = self._extract_registers(src)
                    
                    reads[idx].update(dest_regs)  # dest is both read and written
                    reads[idx].update(src_regs)
                    writes[idx].update(dest_regs)
            
            # Compare instructions
            elif opcode in ['cmp', 'test']:
                # Format: cmp reg1, reg2
                if len(operands) >= 2:
                    reg1 = operands[0].strip()
                    reg2 = operands[1].strip()
                    
                    reg1_regs = self._extract_registers(reg1)
                    reg2_regs = self._extract_registers(reg2)
                    
                    reads[idx].update(reg1_regs)
                    reads[idx].update(reg2_regs)
            
            # Call instruction
            elif opcode in ['call']:
                # Format: call func
                # Assume all general purpose registers are affected
                reads[idx].update(['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi'])
                writes[idx].update(['rax', 'rcx', 'rdx'])
            
            # Jump instructions
            elif opcode.startswith('j'):
                # No direct register effects, but flags are read
                reads[idx].add('flags')
            
            # Increment/decrement
            elif opcode in ['inc', 'dec']:
                # Format: inc reg
                if operands:
                    reg = operands[0].strip()
                    reg_regs = self._extract_registers(reg)
                    
                    reads[idx].update(reg_regs)
                    writes[idx].update(reg_regs)
            
            # Instructions that modify flags
            if opcode not in ['mov', 'push', 'pop', 'call', 'ret']:
                writes[idx].add('flags')
            
            # Conditional instructions that read flags
            if opcode.startswith('j') or opcode in ['cmovz', 'cmovnz', 'cmove', 'cmovne']:
                reads[idx].add('flags')
        
        return reads, writes
    
    def _extract_registers(self, operand):
        """
        Extract register names from an operand.
        
        Args:
            operand: Instruction operand
            
        Returns:
            set: Set of register names
        """
        registers = set()
        
        # Common x86-64 register names
        reg_patterns = [
            'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
            'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
            'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
            'al', 'bl', 'cl', 'dl'
        ]
        
        # Simple register extraction
        for reg in reg_patterns:
            if reg in operand.lower():
                registers.add(reg)
        
        # Handle memory references [reg+offset]
        if '[' in operand and ']' in operand:
            mem_ref = operand[operand.find('[') + 1:operand.find(']')]
            for reg in reg_patterns:
                if reg in mem_ref.lower():
                    registers.add(reg)
        
        return registers
    
    def score(self, results):
        """
        Calculate scores for dead code detection.
        
        Args:
            results: Results from evaluate()
            
        Returns:
            dict: Score metrics
        """
        predictions = results['predictions']
        true_labels = results['true_labels']
        combined_impacts = results['combined_impacts']
        embedding_impacts = results['embedding_impacts']
        data_flow_impacts = results['data_flow_impacts']
        
        # Calculate standard metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Handle potential division by zero
        if sum(predictions) == 0:
            precision = 0
        else:
            precision = precision_score(true_labels, predictions)
            
        if sum(true_labels) == 0:
            recall = 0
        else:
            recall = recall_score(true_labels, predictions)
            
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = f1_score(true_labels, predictions)
        
        # Calculate ROC AUC (using impact scores as the decision function)
        # Invert impact scores since lower impact means higher probability of dead code
        decision_scores = [1.0 - score for score in combined_impacts]
        
        try:
            roc_auc = roc_auc_score(true_labels, decision_scores)
        except ValueError:
            # This can happen if all examples are of one class
            roc_auc = 0.5
        
        # Calculate additional metrics for combined impact
        avg_impact_dead = np.mean([
            score for score, label in zip(combined_impacts, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_impact_live = np.mean([
            score for score, label in zip(combined_impacts, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        combined_impact_separation = avg_impact_live - avg_impact_dead
        
        # Calculate metrics for embedding impact
        avg_emb_impact_dead = np.mean([
            score for score, label in zip(embedding_impacts, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_emb_impact_live = np.mean([
            score for score, label in zip(embedding_impacts, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        emb_impact_separation = avg_emb_impact_live - avg_emb_impact_dead
        
        # Calculate metrics for data flow impact
        avg_df_impact_dead = np.mean([
            score for score, label in zip(data_flow_impacts, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_df_impact_live = np.mean([
            score for score, label in zip(data_flow_impacts, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        df_impact_separation = avg_df_impact_live - avg_df_impact_dead
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_impact_dead': avg_impact_dead,
            'avg_impact_live': avg_impact_live,
            'impact_separation': combined_impact_separation,
            'emb_impact_separation': emb_impact_separation,
            'df_impact_separation': df_impact_separation,
            'data_flow_weight': self.data_flow_weight
        }