"""
Enhanced task for evaluating semantic block equivalence detection.
"""

import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import defaultdict

from .semantic_block import SemanticBlockTask

class StructureAwareBlockTask(SemanticBlockTask):
    """Block equivalence task with control flow structure awareness."""
    
    def __init__(self, threshold=0.75, structure_weight=0.3):
        """
        Initialize the structure-aware block equivalence task.
        
        Args:
            threshold: Similarity threshold for block equivalence detection
            structure_weight: Weight for structural similarity in combined score
        """
        super().__init__(threshold=threshold)
        self.name = "Structure-Aware Semantic Block Equivalence"
        self.description = "Detect different instruction sequences with the same semantic outcome using control flow information"
        self.structure_weight = structure_weight
    
    def evaluate(self, embedding_model, test_data):
        """
        Evaluate the embedding model on semantic block equivalence detection.
        
        Args:
            embedding_model: The embedding model to evaluate
            test_data: Dictionary with keys 'block_pairs' and 'labels'
            
        Returns:
            dict: Evaluation results
        """
        block_pairs = test_data['block_pairs']
        true_labels = test_data['labels']
        
        # Calculate combined similarity scores
        embeddings_sim = []
        structure_sim = []
        combined_sim = []
        
        for block1, block2 in block_pairs:
            # Calculate embedding similarity
            vec1 = embedding_model.transform([block1])[0]
            vec2 = embedding_model.transform([block2])[0]
            
            # Cosine similarity for embeddings
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                emb_sim = 0.0
            else:
                emb_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Calculate structural similarity
            struct_sim = self._calculate_structural_similarity(block1, block2)
            
            # Combined similarity
            combined = (1 - self.structure_weight) * emb_sim + self.structure_weight * struct_sim
            
            embeddings_sim.append(emb_sim)
            structure_sim.append(struct_sim)
            combined_sim.append(combined)
        
        # Predict block equivalence based on threshold
        predictions = [similarity >= self.threshold for similarity in combined_sim]
        
        return {
            'embeddings_similarities': embeddings_sim,
            'structure_similarities': structure_sim,
            'combined_similarities': combined_sim,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def _calculate_structural_similarity(self, block1, block2):
        """
        Calculate structural similarity between two instruction blocks.
        
        Args:
            block1: First instruction block
            block2: Second instruction block
            
        Returns:
            float: Structural similarity score
        """
        # Create control flow graphs
        cfg1 = self._create_control_flow_graph(block1)
        cfg2 = self._create_control_flow_graph(block2)
        
        # Calculate similarity based on graph properties
        if len(cfg1.nodes) == 0 or len(cfg2.nodes) == 0:
            return 0.0
        
        # Compare basic properties
        properties1 = self._extract_graph_properties(cfg1)
        properties2 = self._extract_graph_properties(cfg2)
        
        # Calculate property similarity
        sim_scores = []
        
        # Compare number of nodes
        max_nodes = max(properties1['num_nodes'], properties2['num_nodes'])
        node_sim = 1.0 - abs(properties1['num_nodes'] - properties2['num_nodes']) / max(1, max_nodes)
        sim_scores.append(node_sim)
        
        # Compare number of edges
        max_edges = max(properties1['num_edges'], properties2['num_edges'])
        edge_sim = 1.0 - abs(properties1['num_edges'] - properties2['num_edges']) / max(1, max_edges)
        sim_scores.append(edge_sim)
        
        # Compare number of branch points
        max_branches = max(properties1['num_branches'], properties2['num_branches'])
        branch_sim = 1.0 - abs(properties1['num_branches'] - properties2['num_branches']) / max(1, max_branches)
        sim_scores.append(branch_sim)
        
        # Compare opcode frequencies
        opcode_sim = self._calculate_opcode_frequency_similarity(
            properties1['opcode_freq'], properties2['opcode_freq']
        )
        sim_scores.append(opcode_sim)
        
        # Return average similarity
        return sum(sim_scores) / len(sim_scores)
    
    def _create_control_flow_graph(self, block):
        """
        Create a control flow graph for an instruction block.
        
        Args:
            block: Instruction block
            
        Returns:
            networkx.DiGraph: Control flow graph
        """
        G = nx.DiGraph()
        
        # Add nodes for each instruction
        for i, instruction in enumerate(block):
            G.add_node(i, instruction=instruction)
        
        # Add edges for control flow
        for i in range(len(block) - 1):
            instr = block[i].lower() if isinstance(block[i], str) else block[i].lower()
            
            # Check if instruction is a jump or branch
            is_jump = any(op in instr for op in ['jmp', 'je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb'])
            is_call = 'call' in instr
            is_ret = 'ret' in instr
            
            if is_jump:
                # Add a conditional edge to the next instruction
                G.add_edge(i, i + 1, type='conditional')
                
                # For simplicity, add an edge to a random future instruction
                # In a real implementation, you would parse the jump target
                if i + 2 < len(block):
                    G.add_edge(i, i + 2, type='jump')
            elif is_ret:
                # Return doesn't have outgoing edges
                pass
            elif is_call:
                # Add an edge to the next instruction (assuming call returns)
                G.add_edge(i, i + 1, type='call_return')
            else:
                # Sequential control flow
                G.add_edge(i, i + 1, type='sequential')
        
        return G
    
    def _extract_graph_properties(self, cfg):
        """
        Extract properties from a control flow graph.
        
        Args:
            cfg: Control flow graph
            
        Returns:
            dict: Graph properties
        """
        properties = {}
        
        # Basic properties
        properties['num_nodes'] = len(cfg.nodes)
        properties['num_edges'] = len(cfg.edges)
        
        # Count branch points (nodes with out-degree > 1)
        branch_points = [node for node, degree in cfg.out_degree() if degree > 1]
        properties['num_branches'] = len(branch_points)
        
        # Extract opcode frequencies
        opcode_freq = defaultdict(int)
        for node in cfg.nodes:
            if 'instruction' in cfg.nodes[node]:
                instr = cfg.nodes[node]['instruction']
                if isinstance(instr, str) and instr.strip():
                    opcode = instr.split()[0].lower()
                    opcode_freq[opcode] += 1
        
        properties['opcode_freq'] = opcode_freq
        
        return properties
    
    def _calculate_opcode_frequency_similarity(self, freq1, freq2):
        """
        Calculate similarity between opcode frequency distributions.
        
        Args:
            freq1: First opcode frequency dictionary
            freq2: Second opcode frequency dictionary
            
        Returns:
            float: Similarity score
        """
        # Get all opcodes
        all_opcodes = set(freq1.keys()) | set(freq2.keys())
        
        if not all_opcodes:
            return 0.0
        
        # Calculate cosine similarity between frequency vectors
        vec1 = [freq1.get(op, 0) for op in all_opcodes]
        vec2 = [freq2.get(op, 0) for op in all_opcodes]
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def score(self, results):
        """
        Calculate scores for semantic block equivalence detection.
        
        Args:
            results: Results from evaluate()
            
        Returns:
            dict: Score metrics
        """
        predictions = results['predictions']
        true_labels = results['true_labels']
        combined_similarities = results['combined_similarities']
        embeddings_similarities = results['embeddings_similarities']
        structure_similarities = results['structure_similarities']
        
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
        
        # Calculate ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, combined_similarities)
        except ValueError:
            roc_auc = 0.5
        
        # Calculate additional metrics
        avg_combined_equivalent = np.mean([
            sim for sim, label in zip(combined_similarities, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_combined_different = np.mean([
            sim for sim, label in zip(combined_similarities, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        # Separation based on combined similarity
        combined_separation = avg_combined_equivalent - avg_combined_different
        
        # Calculate embedding-only metrics
        avg_emb_equivalent = np.mean([
            sim for sim, label in zip(embeddings_similarities, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_emb_different = np.mean([
            sim for sim, label in zip(embeddings_similarities, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        # Embedding-only separation
        emb_separation = avg_emb_equivalent - avg_emb_different
        
        # Calculate structure-only metrics
        avg_struct_equivalent = np.mean([
            sim for sim, label in zip(structure_similarities, true_labels) if label
        ]) if any(true_labels) else 0
        
        avg_struct_different = np.mean([
            sim for sim, label in zip(structure_similarities, true_labels) if not label
        ]) if not all(true_labels) else 0
        
        # Structure-only separation
        struct_separation = avg_struct_equivalent - avg_struct_different
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_combined_equivalent': avg_combined_equivalent,
            'avg_combined_different': avg_combined_different,
            'combined_separation': combined_separation,
            'emb_separation': emb_separation,
            'struct_separation': struct_separation,
            'structure_weight': self.structure_weight
        }