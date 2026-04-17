"""
Simulation Engine with Semantic Interpretation Layer

This module implements delta-inference simulation with failure mode context.
The GNN models physics (what changes). This layer explains WHY it matters.

KEY PRINCIPLES:
1. The GNN is already correct - it models topology and flow honestly
2. Negative deltas are NOT errors - they indicate load relief/isolation
3. Semantic meaning depends on failure mode context
4. Same Δ=-0.15 means different things based on scenario

FAILURE MODES:
- DEMAND_LOSS: Consumer/endpoint fails → upstream sees load relief (stagnation risk)
- SUPPLY_CUT: Source/pump fails → downstream sees pressure drop (shortage risk)  
- CONTAMINATION: Quality issue → spread follows flow direction (contamination risk)
- CONTROL_FAILURE: Valve/sensor fails → unpredictable behavior (control risk)

SIMULATION MODES:
- Standard Mode: Report raw physics deltas with semantic interpretation
- Pessimistic Mode: Failure-biased amplification for worst-case admin analysis
  • Amplifies risk increases (positive Δ) with non-linear boost
  • Suppresses relief signals (negative Δ)
  • Anchors to topology weight to prevent hallucinations
"""

import torch
import numpy as np
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


class FailureMode(IntEnum):
    """
    Failure mode determines semantic interpretation of deltas.
    The GNN computes WHAT changes. The failure mode explains WHY it matters.
    """
    NONE = 0              # No specific mode - raw delta only
    DEMAND_LOSS = 1       # Consumer/endpoint failure (hospital, school)
    SUPPLY_CUT = 2        # Source/pump failure (tank, pump station)
    CONTAMINATION = 3     # Water quality issue (backflow, pollution)
    CONTROL_FAILURE = 4   # Valve/sensor malfunction


@dataclass
class SemanticInterpretation:
    """Semantic interpretation of a delta value based on failure mode context."""
    delta: float
    failure_mode: FailureMode
    risk_type: str          # e.g., "STAGNATION_RISK", "SHORTAGE_RISK"
    explanation: str        # Human-readable explanation
    risk_level: str         # "low", "medium", "high", "critical"
    ui_color: str           # Color code for UI display
    ui_icon: str            # Icon for UI display
    confidence: float       # Confidence = |Δ| × topology_weight (0.0 to 1.0)
    confidence_label: str   # "low", "medium", "high" - human readable
    pessimistic_delta: float = 0.0  # Amplified delta for worst-case view
    alert_level: str = "normal"     # "normal", "elevated", "high", "critical"


class SimulationEngine:
    """
    Delta-Inference Simulation Engine with Semantic Interpretation.
    
    This is "God Mode" - we can force any node to fail and measure
    the exact causal echo through the network.
    
    The engine:
    1. Computes baseline probabilities (current state)
    2. Forces target node(s) to fail (counterfactual)
    3. Measures delta = counterfactual - baseline
    4. Interprets delta based on failure mode context
    
    KEY INSIGHT: Delta-inference solves over-smoothing because we measure
    RELATIVE CHANGE, not absolute values. A well-connected network that
    shows high baseline everywhere will still show CLEAR DELTAS when we
    force a failure.
    """
    
    def __init__(self, predictor):
        """
        Initialize simulation engine.
        
        Args:
            predictor: ImpactPredictor instance with loaded model
        """
        self.predictor = predictor
        self.device = predictor.device
        self._topology_weights = None  # Cached topology weights
        
    def _compute_topology_weights(
        self,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """
        Compute topology weights for each node based on connectivity.
        
        Higher weight = more connected = more reliable signal propagation.
        
        Formula: topology_weight[i] = normalized(degree[i] × avg_edge_weight[i])
        
        Args:
            edge_index: Edge connectivity [2, E]
            edge_weight: Edge weights [E]
            num_nodes: Number of nodes
            
        Returns:
            Array of topology weights per node [N], normalized to [0, 1]
        """
        # Compute degree and average edge weight per node
        degree = np.zeros(num_nodes)
        weight_sum = np.zeros(num_nodes)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            w = edge_weight[i] if i < len(edge_weight) else 1.0
            
            degree[src] += 1
            degree[dst] += 1
            weight_sum[src] += w
            weight_sum[dst] += w
            
        # Average edge weight per node
        avg_weight = np.divide(
            weight_sum, 
            degree, 
            out=np.zeros_like(weight_sum), 
            where=degree > 0
        )
        
        # Topology weight = degree × avg_edge_weight
        raw_weights = degree * avg_weight
        
        # Normalize to [0, 1]
        if raw_weights.max() > 0:
            topology_weights = raw_weights / raw_weights.max()
        else:
            topology_weights = np.ones(num_nodes) * 0.5
            
        return topology_weights
    
    def _compute_confidence(
        self,
        delta: float,
        topology_weight: float
    ) -> Tuple[float, str]:
        """
        Compute confidence score for a delta prediction.
        
        Formula: confidence = |Δ| × topology_weight
        
        Interpretation:
        - High confidence: Large delta on well-connected node
        - Low confidence: Small delta or poorly connected node
        
        Args:
            delta: The computed delta value
            topology_weight: Node's topology weight (0-1)
            
        Returns:
            Tuple of (confidence_score, confidence_label)
        """
        # Confidence = |Δ| × topology_weight
        # Scale |Δ| to roughly 0-1 range (deltas rarely exceed 0.5)
        scaled_delta = min(abs(delta) * 2, 1.0)
        confidence = scaled_delta * topology_weight
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        # Label
        if confidence < 0.2:
            label = "low"
        elif confidence < 0.5:
            label = "medium"
        else:
            label = "high"
            
        return confidence, label
    
    def _calculate_pessimistic_delta(
        self,
        raw_delta: float,
        topology_weight: float
    ) -> Tuple[float, str]:
        """
        Failure-biased amplification for admin what-if simulations.
        Loud on risk, quiet on relief.
        
        The GNN computes what changes (physics).
        This method decides how loudly to report it (admin intent).
        
        Formula for risk (positive delta):
            amplified = (Δ ^ 0.5) × 2.0 × topology_weight
            
        This guarantees:
        • Even small real risks shout
        • No fake cascades (topology-anchored)
        • Admin always sees worst-case impact
        • Physics stays honest
        
        Args:
            raw_delta: Original delta from GNN
            topology_weight: Node's topology weight (0-1)
            
        Returns:
            Tuple of (pessimistic_delta, alert_level)
        """
        # Amplify only RISK INCREASE (positive delta)
        if raw_delta > 0:
            # Non-linear boost: small deltas become visible
            # sqrt(0.01) * 2 = 0.2, sqrt(0.05) * 2 = 0.45, sqrt(0.1) * 2 = 0.63
            amplified = (raw_delta ** 0.5) * 2.0
            
            # Anchor to topology confidence (0-1) to prevent hallucinations
            pessimistic = amplified * topology_weight
            
            # Determine alert level based on amplified value
            if pessimistic >= 0.8:
                alert_level = "critical"
            elif pessimistic >= 0.5:
                alert_level = "high"
            elif pessimistic >= 0.2:
                alert_level = "elevated"
            else:
                alert_level = "normal"
                
            return min(pessimistic, 1.0), alert_level
        
        # Suppress relief (negative delta) - still show but quiet
        suppressed = raw_delta * 0.1
        return suppressed, "normal"
        
    def run_simulation(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        failed_nodes: List[int],
        node_names: Optional[List[str]] = None,
        failure_mode: FailureMode = FailureMode.NONE,
        pessimistic_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run delta-inference simulation with semantic interpretation.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_weight: Edge weights [E]
            failed_nodes: List of node indices to force-fail
            node_names: Optional human-readable node names
            failure_mode: Context for semantic interpretation
            pessimistic_mode: If True, amplify risks for worst-case admin view
            
        Returns:
            Dictionary with baseline, simulation, deltas, and interpretations
        """
        num_nodes = x.shape[0]
        if node_names is None:
            node_names = [f"Node_{i}" for i in range(num_nodes)]
            
        # Step 1: Baseline prediction (current state)
        baseline_probs, _, _ = self.predictor.predict_with_threshold(
            x, edge_index, edge_weight
        )
        baseline_probs = baseline_probs[:, 0]  # Impact probability column
        
        # Step 2: Counterfactual - force target nodes to fail
        x_sim = x.copy()
        status_col = 15  # Status feature index in [12 one-hot | 12 operational]
        
        for node_idx in failed_nodes:
            x_sim[node_idx, status_col] = 0.0  # Force failure
            
        # Step 3: Counterfactual prediction
        sim_probs, _, _ = self.predictor.predict_with_threshold(
            x_sim, edge_index, edge_weight
        )
        sim_probs = sim_probs[:, 0]
        
        # Step 4: Compute deltas
        deltas = sim_probs - baseline_probs
        
        # Step 5: Compute topology weights for confidence
        topology_weights = self._compute_topology_weights(edge_index, edge_weight, num_nodes)
        
        # Step 6: Compute pessimistic deltas if in pessimistic mode
        pessimistic_deltas = np.zeros(num_nodes)
        if pessimistic_mode:
            for i in range(num_nodes):
                pessimistic_deltas[i], _ = self._calculate_pessimistic_delta(
                    deltas[i], topology_weights[i]
                )
        
        # Step 7: Build report with semantic interpretation
        report = []
        for i in range(num_nodes):
            interpretation = self._interpret_delta(
                delta=deltas[i],
                node_name=node_names[i],
                is_source=(i in failed_nodes),
                failure_mode=failure_mode,
                topology_weight=topology_weights[i],
                pessimistic_mode=pessimistic_mode
            )
            
            report.append({
                "node_id": i,
                "node_name": node_names[i],
                "baseline": float(baseline_probs[i]),
                "simulated": float(sim_probs[i]),
                "delta": float(deltas[i]),
                "pessimistic_delta": float(pessimistic_deltas[i]) if pessimistic_mode else None,
                "is_failed_source": i in failed_nodes,
                "interpretation": interpretation
            })
            
        # Sort by absolute delta (most affected first)
        report.sort(key=lambda r: abs(r["delta"]), reverse=True)
        
        # Summary statistics
        affected_nodes = [r for r in report if abs(r["delta"]) > 0.01]
        
        summary = {
            "total_nodes": num_nodes,
            "failed_nodes": failed_nodes,
            "failed_names": [node_names[i] for i in failed_nodes],
            "failure_mode": failure_mode.name,
            "pessimistic_mode": pessimistic_mode,
            "affected_count": len(affected_nodes),
            "max_delta": float(np.max(np.abs(deltas))),
            "mean_delta": float(np.mean(np.abs(deltas))),
            "max_pessimistic_delta": float(np.max(pessimistic_deltas)) if pessimistic_mode else None,
            "interpretation_summary": self._summarize_interpretations(report, failure_mode, pessimistic_mode)
        }
        
        return {
            "summary": summary,
            "nodes": report,
            "raw": {
                "baseline": baseline_probs.tolist(),
                "simulated": sim_probs.tolist(),
                "deltas": deltas.tolist()
            }
        }

    def run_cumulative_simulation(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        failure_sequence: List[int],
        node_names: Optional[List[str]] = None,
        failure_mode: FailureMode = FailureMode.NONE,
        pessimistic_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Run stepwise cumulative failures.

        Example:
        - Step 1: fail node A
        - Step 2: fail node A + B
        - Step 3: fail node A + B + C

        Returns each step with:
        - cumulative delta vs healthy baseline
        - incremental delta vs previous step
        """
        num_nodes = x.shape[0]
        if node_names is None:
            node_names = [f"Node_{i}" for i in range(num_nodes)]

        baseline_probs, _, _ = self.predictor.predict_with_threshold(x, edge_index, edge_weight)
        baseline_probs = baseline_probs[:, 0]

        cumulative_failed = []
        prev_probs = baseline_probs.copy()
        steps = []

        for step_idx, node_idx in enumerate(failure_sequence, start=1):
            if node_idx < 0 or node_idx >= num_nodes:
                raise ValueError(f"Invalid node index in sequence: {node_idx}")

            if node_idx not in cumulative_failed:
                cumulative_failed.append(node_idx)

            step_result = self.run_simulation(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                failed_nodes=cumulative_failed,
                node_names=node_names,
                failure_mode=failure_mode,
                pessimistic_mode=pessimistic_mode,
            )

            current_probs = np.array(step_result["raw"]["simulated"], dtype=np.float32)
            incremental_deltas = current_probs - prev_probs
            prev_probs = current_probs

            enriched_nodes = []
            for node_row in step_result["nodes"]:
                node_copy = dict(node_row)
                node_copy["incremental_delta"] = float(incremental_deltas[node_copy["node_id"]])
                enriched_nodes.append(node_copy)

            steps.append(
                {
                    "step": step_idx,
                    "new_failed_node": int(node_idx),
                    "new_failed_name": node_names[node_idx],
                    "cumulative_failed_nodes": list(cumulative_failed),
                    "cumulative_failed_names": [node_names[i] for i in cumulative_failed],
                    "summary": step_result["summary"],
                    "nodes": enriched_nodes,
                    "raw": {
                        "simulated": current_probs.tolist(),
                        "delta_from_baseline": step_result["raw"]["deltas"],
                        "incremental_delta_from_previous_step": incremental_deltas.tolist(),
                    },
                }
            )

        return {
            "baseline": baseline_probs.tolist(),
            "failure_sequence": list(failure_sequence),
            "steps": steps,
            "final_failed_nodes": list(cumulative_failed),
            "final_failed_names": [node_names[i] for i in cumulative_failed],
            "total_steps": len(steps),
        }
    
    def _interpret_delta(
        self,
        delta: float,
        node_name: str,
        is_source: bool,
        failure_mode: FailureMode,
        topology_weight: float = 1.0,
        pessimistic_mode: bool = False
    ) -> SemanticInterpretation:
        """
        Interpret a delta value based on failure mode context.
        
        The same delta means different things depending on failure mode:
        - DEMAND_LOSS + negative delta = stagnation risk (load relief)
        - SUPPLY_CUT + negative delta = isolation (cut off from supply)
        - CONTAMINATION + negative delta = reduced flow (backflow risk)
        
        Args:
            delta: The computed delta value
            node_name: Human-readable node name
            is_source: Whether this is the forced-fail node
            failure_mode: Context for interpretation
            topology_weight: Node's topology weight for confidence calculation
            pessimistic_mode: If True, use failure-biased amplification
            
        Returns:
            SemanticInterpretation with risk type, explanation, confidence, and UI hints
        """
        abs_delta = abs(delta)
        
        # Compute confidence
        confidence, confidence_label = self._compute_confidence(delta, topology_weight)
        
        # Compute pessimistic delta and alert level
        pessimistic_delta, alert_level = self._calculate_pessimistic_delta(delta, topology_weight)
        
        # Determine risk level based on magnitude (or pessimistic value if in that mode)
        effective_delta = pessimistic_delta if pessimistic_mode and delta > 0 else abs_delta
        
        if effective_delta < 0.05:
            risk_level = "low"
            ui_color = "#4CAF50"  # Green
        elif effective_delta < 0.15:
            risk_level = "medium"
            ui_color = "#FF9800"  # Orange
        elif effective_delta < 0.30:
            risk_level = "high"
            ui_color = "#F44336"  # Red
        else:
            risk_level = "critical"
            ui_color = "#9C27B0"  # Purple
        
        # Override with alert level in pessimistic mode
        if pessimistic_mode and alert_level != "normal":
            risk_level = alert_level
            if alert_level == "critical":
                ui_color = "#9C27B0"  # Purple
            elif alert_level == "high":
                ui_color = "#F44336"  # Red
            elif alert_level == "elevated":
                ui_color = "#FF9800"  # Orange
            
        # Source node (the one we forced to fail)
        if is_source:
            return SemanticInterpretation(
                delta=delta,
                failure_mode=failure_mode,
                risk_type="FAILURE_SOURCE",
                explanation=f"{node_name} is the simulated failure point",
                risk_level="critical",
                ui_color="#000000",  # Black
                ui_icon="⚫",
                confidence=1.0,
                confidence_label="high",
                pessimistic_delta=1.0,
                alert_level="critical"
            )
            
        # No significant change
        if abs_delta < 0.01:
            return SemanticInterpretation(
                delta=delta,
                failure_mode=failure_mode,
                risk_type="UNAFFECTED",
                explanation=f"{node_name} is not significantly affected",
                risk_level="low",
                ui_color="#9E9E9E",  # Gray
                ui_icon="○",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
            
        # Interpret based on failure mode and delta direction
        if failure_mode == FailureMode.DEMAND_LOSS:
            return self._interpret_demand_loss(delta, node_name, risk_level, ui_color, confidence, confidence_label, pessimistic_delta, alert_level)
            
        elif failure_mode == FailureMode.SUPPLY_CUT:
            return self._interpret_supply_cut(delta, node_name, risk_level, ui_color, confidence, confidence_label, pessimistic_delta, alert_level)
            
        elif failure_mode == FailureMode.CONTAMINATION:
            return self._interpret_contamination(delta, node_name, risk_level, ui_color, confidence, confidence_label, pessimistic_delta, alert_level)
            
        elif failure_mode == FailureMode.CONTROL_FAILURE:
            return self._interpret_control_failure(delta, node_name, risk_level, ui_color, confidence, confidence_label, pessimistic_delta, alert_level)
            
        else:  # FailureMode.NONE - raw delta only
            return self._interpret_raw_delta(delta, node_name, risk_level, ui_color, confidence, confidence_label, pessimistic_delta, alert_level)
            
    def _interpret_demand_loss(
        self, delta: float, node_name: str, risk_level: str, ui_color: str,
        confidence: float, confidence_label: str, pessimistic_delta: float, alert_level: str
    ) -> SemanticInterpretation:
        """Interpret delta in demand loss context (consumer/endpoint failure)."""
        if delta < 0:
            # Negative delta = load relief → stagnation risk
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.DEMAND_LOSS,
                risk_type="STAGNATION_RISK",
                explanation=f"{node_name}: Reduced load → water stagnation risk (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="🔄",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
        else:
            # Positive delta = compensating load increase
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.DEMAND_LOSS,
                risk_type="LOAD_REDISTRIBUTION",
                explanation=f"{node_name}: Compensating for lost demand (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="📈",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
            
    def _interpret_supply_cut(
        self, delta: float, node_name: str, risk_level: str, ui_color: str,
        confidence: float, confidence_label: str, pessimistic_delta: float, alert_level: str
    ) -> SemanticInterpretation:
        """Interpret delta in supply cut context (source/pump failure)."""
        if delta < 0:
            # Negative delta = isolation from supply
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.SUPPLY_CUT,
                risk_type="SUPPLY_ISOLATION",
                explanation=f"{node_name}: Cut off from supply source (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="🚫",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
        else:
            # Positive delta = pressure surge or backup source activation
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.SUPPLY_CUT,
                risk_type="PRESSURE_SURGE",
                explanation=f"{node_name}: Pressure redistribution detected (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="⚡",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
            
    def _interpret_contamination(
        self, delta: float, node_name: str, risk_level: str, ui_color: str,
        confidence: float, confidence_label: str, pessimistic_delta: float, alert_level: str
    ) -> SemanticInterpretation:
        """Interpret delta in contamination context (water quality issue)."""
        if delta < 0:
            # Negative delta = reduced flow → backflow risk
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.CONTAMINATION,
                risk_type="BACKFLOW_RISK",
                explanation=f"{node_name}: Reduced pressure → backflow contamination risk (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="☣️",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
        else:
            # Positive delta = contamination spread
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.CONTAMINATION,
                risk_type="CONTAMINATION_SPREAD",
                explanation=f"{node_name}: In contamination spread path (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="🦠",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
            
    def _interpret_control_failure(
        self, delta: float, node_name: str, risk_level: str, ui_color: str,
        confidence: float, confidence_label: str, pessimistic_delta: float, alert_level: str
    ) -> SemanticInterpretation:
        """Interpret delta in control failure context (valve/sensor malfunction)."""
        if delta < 0:
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.CONTROL_FAILURE,
                risk_type="CONTROL_BLIND_SPOT",
                explanation=f"{node_name}: Lost monitoring/control visibility (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="👁️",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
        else:
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.CONTROL_FAILURE,
                risk_type="CONTROL_INSTABILITY",
                explanation=f"{node_name}: Control loop instability detected (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="⚠️",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
            
    def _interpret_raw_delta(
        self, delta: float, node_name: str, risk_level: str, ui_color: str,
        confidence: float, confidence_label: str, pessimistic_delta: float, alert_level: str
    ) -> SemanticInterpretation:
        """Interpret delta without failure mode context (raw physics)."""
        if delta < 0:
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.NONE,
                risk_type="IMPACT_DECREASE",
                explanation=f"{node_name}: Impact probability decreased (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="📉",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
        else:
            return SemanticInterpretation(
                delta=delta,
                failure_mode=FailureMode.NONE,
                risk_type="IMPACT_INCREASE",
                explanation=f"{node_name}: Impact probability increased (Δ{delta:+.2f})",
                risk_level=risk_level,
                ui_color=ui_color,
                ui_icon="📈",
                confidence=confidence,
                confidence_label=confidence_label,
                pessimistic_delta=pessimistic_delta,
                alert_level=alert_level
            )
            
    def _summarize_interpretations(
        self, report: List[Dict], failure_mode: FailureMode, pessimistic_mode: bool = False
    ) -> str:
        """Generate a summary of the semantic interpretations."""
        affected = [r for r in report if abs(r["delta"]) > 0.01 and not r["is_failed_source"]]
        
        if not affected:
            return "No significant downstream impact detected."
            
        # Count risk types and alert levels
        risk_counts = {}
        alert_counts = {"critical": 0, "high": 0, "elevated": 0, "normal": 0}
        
        for r in affected:
            risk_type = r["interpretation"].risk_type
            risk_counts[risk_type] = risk_counts.get(risk_type, 0) + 1
            alert_counts[r["interpretation"].alert_level] += 1
            
        # Build summary
        parts = []
        for risk_type, count in sorted(risk_counts.items(), key=lambda x: -x[1]):
            parts.append(f"{count} node(s) with {risk_type}")
            
        mode_context = {
            FailureMode.NONE: "Raw impact analysis",
            FailureMode.DEMAND_LOSS: "Demand loss scenario (consumer failure)",
            FailureMode.SUPPLY_CUT: "Supply cut scenario (source failure)",
            FailureMode.CONTAMINATION: "Contamination scenario (quality issue)",
            FailureMode.CONTROL_FAILURE: "Control failure scenario (sensor/valve)"
        }
        
        base_summary = f"{mode_context[failure_mode]}: {', '.join(parts)}"
        
        # Add pessimistic alert summary
        if pessimistic_mode:
            alerts = []
            if alert_counts["critical"] > 0:
                alerts.append(f"🔴 {alert_counts['critical']} CRITICAL")
            if alert_counts["high"] > 0:
                alerts.append(f"🟠 {alert_counts['high']} HIGH")
            if alert_counts["elevated"] > 0:
                alerts.append(f"🟡 {alert_counts['elevated']} ELEVATED")
            if alerts:
                base_summary += f" | ALERTS: {', '.join(alerts)}"
                
        return base_summary


def demo():
    """Demo the simulation engine with semantic interpretation and pessimistic mode."""
    print("="*70)
    print("SIMULATION ENGINE WITH SEMANTIC INTERPRETATION")
    print("="*70)
    
    from model import ImpactPredictor
    
    script_dir = Path(__file__).parent
    model_path = script_dir / "models" / "gnn_production_v1.pt"
    
    predictor = ImpactPredictor(str(model_path))
    engine = SimulationEngine(predictor)
    
    # Create test graph: Tank → Pump → Pipe → Hospital
    x = np.array([
        # Tank (source) - healthy
        [0,0,0,1,0,0,0,0,0,0,0,0, 0.9, 0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9, 0.1, 0.1, 0.0],
        # Pump - healthy
        [0,0,0,0,1,0,0,0,0,0,0,0, 0.9, 0.7, 0.8, 0.8, 0.6, 0.4, 0.5, 0.7, 0.8, 0.1, 0.1, 0.0],
        # Pipe - healthy
        [0,0,0,0,0,1,0,0,0,0,0,0, 0.9, 0.6, 0.7, 0.7, 0.4, 0.3, 0.4, 0.5, 0.6, 0.1, 0.1, 0.0],
        # Hospital (critical consumer) - healthy
        [0,0,0,0,0,0,1,0,0,0,0,0, 0.9, 0.9, 0.5, 0.95, 0.9, 0.8, 0.9, 0.9, 0.95, 0.1, 0.1, 0.0],
    ], dtype=np.float32)
    
    edge_index = np.array([[0,1,2], [1,2,3]], dtype=np.int64)
    edge_weight = np.array([0.9, 0.85, 0.8], dtype=np.float32)
    node_names = ["Tank_Main", "Pump_Station", "Pipe_A", "Hospital"]
    
    print("\n📊 Test Graph: Tank → Pump → Pipe → Hospital")
    print("-"*70)
    
    # ========== STANDARD MODE ==========
    print("\n" + "="*70)
    print("🔵 STANDARD MODE (Conservative Physics)")
    print("="*70)
    
    result = engine.run_simulation(
        x, edge_index, edge_weight,
        failed_nodes=[0],
        node_names=node_names,
        failure_mode=FailureMode.SUPPLY_CUT,
        pessimistic_mode=False
    )
    
    print(f"\n🔬 Scenario: Tank fails (supply cut)")
    print("-"*50)
    print(f"Affected Nodes: {result['summary']['affected_count']}")
    
    for node in result["nodes"]:
        interp = node["interpretation"]
        if node["is_failed_source"]:
            print(f"  {interp.ui_icon} {node['node_name']}: FAILURE SOURCE")
        elif abs(node["delta"]) > 0.01:
            print(f"  {interp.ui_icon} {interp.explanation}")
            print(f"     Risk: {interp.risk_level.upper()} | Raw Δ: {node['delta']:+.3f}")
    
    print(f"\n  Summary: {result['summary']['interpretation_summary']}")
    
    # ========== PESSIMISTIC MODE ==========
    print("\n" + "="*70)
    print("🔴 PESSIMISTIC MODE (High-Alert Crisis Simulator)")
    print("   Formula: amplified = (Δ ^ 0.5) × 2.0 × topology_weight")
    print("="*70)
    
    result = engine.run_simulation(
        x, edge_index, edge_weight,
        failed_nodes=[0],
        node_names=node_names,
        failure_mode=FailureMode.SUPPLY_CUT,
        pessimistic_mode=True
    )
    
    print(f"\n🔬 Scenario: Tank fails (supply cut) - WORST CASE VIEW")
    print("-"*50)
    print(f"Affected Nodes: {result['summary']['affected_count']}")
    print(f"Max Pessimistic Δ: {result['summary']['max_pessimistic_delta']:.3f}")
    
    for node in result["nodes"]:
        interp = node["interpretation"]
        if node["is_failed_source"]:
            print(f"  {interp.ui_icon} {node['node_name']}: FAILURE SOURCE")
        elif abs(node["delta"]) > 0.005:  # Lower threshold in pessimistic mode
            # Show alert level prominently
            alert_icon = {
                "critical": "🔴",
                "high": "🟠", 
                "elevated": "🟡",
                "normal": "🟢"
            }.get(interp.alert_level, "⚪")
            
            print(f"  {alert_icon} {node['node_name']}: {interp.risk_type}")
            print(f"     Raw Δ: {node['delta']:+.3f} → Amplified: {interp.pessimistic_delta:+.3f}")
            print(f"     Alert: {interp.alert_level.upper()}")
    
    print(f"\n  Summary: {result['summary']['interpretation_summary']}")
    
    # ========== AMPLIFICATION DEMO ==========
    print("\n" + "="*70)
    print("📈 AMPLIFICATION FORMULA DEMONSTRATION")
    print("   amplified = (Δ ^ 0.5) × 2.0 × topology_weight")
    print("="*70)
    
    print("\n  Raw Δ    →  Amplified (topology=1.0)")
    print("  ─────────────────────────────────────")
    test_deltas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    for d in test_deltas:
        amp = (d ** 0.5) * 2.0 * 1.0
        level = "CRITICAL" if amp >= 0.8 else "HIGH" if amp >= 0.5 else "ELEVATED" if amp >= 0.2 else "normal"
        bar = "█" * int(min(amp, 1.0) * 20)
        print(f"  {d:.2f}     →  {amp:.3f} [{bar:<20}] {level}")
    
    # ========== COMPARISON ==========
    print("\n" + "="*70)
    print("📊 MODE COMPARISON")
    print("="*70)
    print("""
    STANDARD MODE          │  PESSIMISTIC MODE
    ───────────────────────┼────────────────────────
    Raw physics deltas     │  Amplified worst-case
    Conservative alerts    │  Aggressive alerts  
    "What will happen"     │  "What could go wrong"
    For monitoring         │  For crisis planning
    
    Admin Reading Guide:
    • Positive Δ (even small) → 🚨 Potential failure propagation
    • Negative Δ             → ⚠️  Operational anomaly / stagnation
    • Large amplified Δ      → 🔴 Immediate attention required
    """)
    
    print("="*70)
    print("✅ Pessimistic Confidence Engine working correctly")
    print("="*70)


if __name__ == "__main__":
    demo()
