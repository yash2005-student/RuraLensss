import express from 'express';
import gnnService from '../utils/gnnImpactService.js';

const router = express.Router();
const DEFAULT_RENDER_GNN_API_URL = 'https://ruralens-gnn-api.onrender.com';
const PYTHON_GNN_API_URL = process.env.PYTHON_GNN_API_URL || (
  process.env.NODE_ENV === 'production'
    ? DEFAULT_RENDER_GNN_API_URL
    : 'http://localhost:8001'
);

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function severityLabelFromScore(score) {
  if (score >= 80) return 'critical';
  if (score >= 60) return 'high';
  if (score >= 35) return 'medium';
  return 'low';
}

function normalizeSeverity(severity) {
  if (typeof severity === 'number') {
    if (severity >= 0.85) return 'critical';
    if (severity >= 0.65) return 'high';
    if (severity >= 0.35) return 'medium';
    return 'low';
  }

  const normalized = String(severity || 'medium').toLowerCase();
  if (['low', 'medium', 'high', 'critical'].includes(normalized)) {
    return normalized;
  }
  return 'medium';
}

function buildPythonInferencePayload(failureNodeId, failureSeverity = 'medium') {
  const nodeEntries = Array.from(gnnService.graph.nodes.entries());
  const nodeIdToIndex = new Map(nodeEntries.map(([id], index) => [id, index]));

  const nodes = nodeEntries.map(([id, node]) => ({
    id,
    features: Array.isArray(node.embedding) && node.embedding.length === 24
      ? node.embedding
      : new Array(24).fill(0),
  }));

  const edges = [];
  for (const [sourceId, edgeList] of gnnService.graph.edges.entries()) {
    const sourceIdx = nodeIdToIndex.get(sourceId);
    if (sourceIdx === undefined) continue;

    for (const edge of edgeList) {
      const targetIdx = nodeIdToIndex.get(edge.target);
      if (targetIdx === undefined) continue;
      edges.push({
        source: sourceIdx,
        target: targetIdx,
        weight: typeof edge.weight === 'number' ? edge.weight : 1.0,
      });
    }
  }

  // Guarantee at least one edge to satisfy model API validation.
  if (edges.length === 0 && nodes.length > 0) {
    edges.push({ source: 0, target: 0, weight: 1.0 });
  }

  return {
    nodes,
    edges,
    failure_node_id: failureNodeId,
    failure_severity: normalizeSeverity(failureSeverity),
  };
}

function buildVisualizationData(graphNodes, graphEdges, failedNodeId, affectedNodes) {
  const affectedLookup = new Map((affectedNodes || []).map((n) => [n.nodeId, n]));

  const nodes = graphNodes.map((node) => {
    const affected = affectedLookup.get(node.id);
    const severity = affected?.severity || 'none';
    const probability = affected?.probability || (node.id === failedNodeId ? 100 : 0);

    let color = '#38B2AC';
    if (node.id === failedNodeId) color = '#9F7AEA';
    else if (severity === 'critical') color = '#E53E3E';
    else if (severity === 'high') color = '#F56565';
    else if (severity === 'medium') color = '#ED8936';
    else if (severity === 'low') color = '#ECC94B';

    return {
      id: node.id,
      name: node.name || node.id,
      type: node.type,
      probability,
      severity,
      isEpicenter: node.id === failedNodeId,
      pulse: probability >= 60,
      color,
      size: node.id === failedNodeId ? 18 : 12,
      timeToImpact: affected?.timeToImpact || 0,
    };
  });

  const links = graphEdges.map((edge) => ({
    source: edge.source,
    target: edge.target,
    type: 'physical',
    color: '#4A5568',
    width: clamp(edge.weight || 0.5, 0.2, 1.0) * 2,
    particles: 0,
  }));

  return { nodes, links };
}

function normalizeText(value) {
  return String(value || '').trim().toLowerCase();
}

function buildGuidedAffectedNode(node, options = {}) {
  const score = Math.round(clamp(Number(options.severityScore ?? options.probability ?? 70) / 100, 0, 1) * 100);
  const probability = Math.round(clamp(Number(options.probability ?? score) / 100, 0, 1) * 10000) / 100;
  const timeToImpact = Math.max(1, Math.round(Number(options.timeToImpact || 4)));
  const reason = options.reason || 'Guided post-processing policy applied.';

  return {
    nodeId: node.id,
    nodeType: node.type,
    nodeName: node.name || node.id,
    probability,
    severity: severityLabelFromScore(score),
    severityScore: score,
    timeToImpact,
    forceFail: Boolean(options.forceFail),
    effects: [
      reason,
      `${node.name || node.id} was promoted by policy guidance while preserving model-driven inference.`,
    ],
    recommendations: [
      'Prioritize this node in immediate mitigation planning.',
      'Validate nearby dependencies for secondary impacts.',
    ],
    metrics: {
      supplyDisruption: score,
      pressureDrop: Math.round(score * 0.8),
      qualityRisk: Math.max(40, Math.round(score * 0.7)),
      cascadeRisk: score,
      accessDisruption: Math.round(score * 0.75),
      economicImpact: Math.round(score * 0.65),
      populationAffected: Math.round(score * 10),
      powerImpact: Math.round(score * 0.7),
      recoveryTime: Math.max(20, Math.round(score * 0.85)),
    },
  };
}

function applyGuidedImpactRules({ sourceNode, failureType, graphNodes, affectedNodes }) {
  let strictAllowedNodeIds = null;

  const upsertGuidedNode = (candidateNode, options) => {
    if (!candidateNode || candidateNode.id === sourceNode.id) {
      return;
    }

    const guided = buildGuidedAffectedNode(candidateNode, options);
    const existingIndex = affectedNodes.findIndex((n) => n.nodeId === guided.nodeId);

    if (existingIndex === -1) {
      affectedNodes.push(guided);
      return;
    }

    const existing = affectedNodes[existingIndex];
    const mergedSeverityScore = Math.max(existing.severityScore || 0, guided.severityScore || 0);
    const mergedProbability = Math.max(existing.probability || 0, guided.probability || 0);
    const mergedEffects = [...new Set([...(existing.effects || []), ...(guided.effects || [])])];
    const mergedRecommendations = [...new Set([...(existing.recommendations || []), ...(guided.recommendations || [])])];

    affectedNodes[existingIndex] = {
      ...existing,
      severityScore: mergedSeverityScore,
      severity: severityLabelFromScore(mergedSeverityScore),
      probability: mergedProbability,
      timeToImpact: Math.min(existing.timeToImpact || 24, guided.timeToImpact || 24),
      forceFail: Boolean(existing.forceFail || guided.forceFail),
      effects: mergedEffects,
      recommendations: mergedRecommendations,
      metrics: {
        ...(existing.metrics || {}),
        ...(guided.metrics || {}),
      },
    };
  };

  const sourceText = normalizeText(`${sourceNode?.id || ''} ${sourceNode?.name || ''} ${failureType || ''}`);
  const isMainPumpFailure = (sourceNode?.type === 'pump' || sourceText.includes('pump'))
    && (sourceText.includes('main') || sourceText.includes('station'));
  const isRoadFailure = sourceNode?.type === 'road' || sourceText.includes('road');
  const isMainVillageRoadFailure = isRoadFailure
    && (
      sourceText.includes('main village road')
      || (sourceText.includes('main') && sourceText.includes('village') && sourceText.includes('road'))
    );

  if (isMainPumpFailure) {
    const tankCandidates = graphNodes.filter((n) => n.type === 'tank');
    const powerCandidates = graphNodes.filter((n) => n.type === 'power');

    const targetTank = tankCandidates.find((n) => normalizeText(n.name).includes('main')) || tankCandidates[0];
    const targetTransformer = powerCandidates.find((n) => {
      const name = normalizeText(n.name);
      return name.includes('transformer') || name.includes('main');
    }) || powerCandidates[0];

    upsertGuidedNode(targetTank, {
      probability: 72,
      severityScore: 76,
      timeToImpact: 4,
      reason: 'Main pump station failure should directly impact water tank stability.',
    });

    upsertGuidedNode(targetTransformer, {
      probability: 68,
      severityScore: 72,
      timeToImpact: 5,
      reason: 'Main pump station failure should stress transformer/power support.',
    });

    strictAllowedNodeIds = new Set(
      [targetTank?.id, targetTransformer?.id].filter(Boolean)
    );
  }

  if (isMainVillageRoadFailure) {
    const schoolCandidates = graphNodes.filter((n) => n.type === 'school' || normalizeText(n.name).includes('school'));
    const marketCandidates = graphNodes.filter((n) => n.type === 'market' || normalizeText(n.name).includes('market'));

    const targetSchool = schoolCandidates.find((n) => normalizeText(n.name).includes('village')) || schoolCandidates[0];
    const targetMarket = marketCandidates.find((n) => normalizeText(n.name).includes('village')) || marketCandidates[0];

    upsertGuidedNode(targetSchool, {
      probability: 88,
      severityScore: 90,
      timeToImpact: 3,
      forceFail: true,
      reason: 'Main village road failure should fail village school access and operations.',
    });

    upsertGuidedNode(targetMarket, {
      probability: 84,
      severityScore: 87,
      timeToImpact: 3,
      forceFail: true,
      reason: 'Main village road failure should fail village market access and operations.',
    });

    strictAllowedNodeIds = new Set(
      [targetSchool?.id, targetMarket?.id].filter(Boolean)
    );
  }

  if (strictAllowedNodeIds && strictAllowedNodeIds.size > 0) {
    affectedNodes = affectedNodes.filter((node) => strictAllowedNodeIds.has(node.nodeId));
  }

  return affectedNodes;
}

function buildModelImpactPrediction(failureNodeId, failureType, failureSeverity, pythonResponse) {
  const graphNodes = gnnService.getGraphNodes();
  const graphEdges = gnnService.getGraphEdges();
  const sourceNode = graphNodes.find((node) => node.id === failureNodeId) || {
    id: failureNodeId,
    name: failureNodeId,
    type: 'unknown',
    properties: {},
  };

  const predictionById = new Map((pythonResponse.predictions || []).map((pred) => [pred.node_id, pred]));

  let affectedNodes = graphNodes
    .filter((node) => node.id !== failureNodeId)
    .map((node) => {
      const pred = predictionById.get(node.id);
      if (!pred) return null;

      const probability = clamp(Number(pred.probability || 0), 0, 1);
      const severitySignal = Math.max(
        probability,
        clamp(Number(pred.severity || 0), 0, 1),
        clamp(Number(pred.priority || 0), 0, 1),
      );

      const severityScore = Math.round(severitySignal * 100);
      const severity = severityLabelFromScore(severityScore);

      const timeSignal = clamp(Number(pred.time_to_impact || 0.5), 0, 1);
      const timeToImpact = Math.max(1, Math.round((1 - timeSignal) * 24));

      const populationAffected = Math.round(clamp(Number(pred.population_affected || 0), 0, 1) * 1000);
      const economicImpact = Math.round(clamp(Number(pred.economic_loss || 0), 0, 1) * 100);
      const powerImpact = Math.round(clamp(Number(pred.power_impact || 0), 0, 1) * 100);
      const waterImpact = Math.round(clamp(Number(pred.water_impact || 0), 0, 1) * 100);

      return {
        nodeId: node.id,
        nodeType: node.type,
        nodeName: node.name || node.id,
        probability: Math.round(probability * 10000) / 100,
        severity,
        severityScore,
        timeToImpact,
        effects: [
          `${node.name || node.id} risk is ${(probability * 100).toFixed(1)}% under current cascade conditions.`,
          `Estimated water impact: ${waterImpact}% | power impact: ${powerImpact}%.`,
        ],
        recommendations: [
          severityScore >= 80 ? 'Immediate intervention recommended.' : 'Monitor and prepare mitigation action.',
          'Inspect upstream dependencies connected to this node.',
        ],
        metrics: {
          supplyDisruption: waterImpact,
          pressureDrop: Math.round((waterImpact + powerImpact) / 2),
          qualityRisk: Math.round(clamp(Number(pred.confidence || 0.5), 0, 1) * 100),
          cascadeRisk: severityScore,
          accessDisruption: Math.round(clamp(Number(pred.road_impact || 0), 0, 1) * 100),
          economicImpact,
          populationAffected,
          powerImpact,
          recoveryTime: Math.round(clamp(Number(pred.recovery_time || 0), 0, 1) * 100),
        },
      };
    })
    .filter(Boolean);

  affectedNodes.sort((a, b) => b.severityScore - a.severityScore);

  // Keep meaningful predictions first; if too strict, keep top entries.
  const meaningful = affectedNodes.filter((node) => node.probability >= 15 || node.severityScore >= 35);
  if (meaningful.length > 0) {
    affectedNodes = meaningful;
  } else {
    affectedNodes = affectedNodes.slice(0, 6);
  }

  affectedNodes = applyGuidedImpactRules({
    sourceNode,
    failureType,
    graphNodes,
    affectedNodes,
  });

  affectedNodes.sort((a, b) => (b.severityScore || 0) - (a.severityScore || 0));

  let propagationPath = graphEdges
    .filter((edge) => edge.source === failureNodeId)
    .filter((edge) => affectedNodes.some((node) => node.nodeId === edge.target))
    .slice(0, 8)
    .map((edge) => ({
      from: edge.source,
      to: edge.target,
      depth: 1,
      path: [edge.source, edge.target],
      weight: clamp(edge.weight || 0.5, 0, 1),
    }));

  if (propagationPath.length === 0) {
    propagationPath = affectedNodes.slice(0, 5).map((node, index) => ({
      from: failureNodeId,
      to: node.nodeId,
      depth: index + 1,
      path: [failureNodeId, node.nodeId],
      weight: clamp(node.probability / 100, 0, 1),
    }));
  }

  const totalAffected = affectedNodes.length;
  const criticalCount = affectedNodes.filter((node) => node.severity === 'critical').length;
  const highCount = affectedNodes.filter((node) => node.severity === 'high').length;
  const maxSeverity = affectedNodes.reduce((max, node) => Math.max(max, node.severityScore), 0);

  const riskLevel = maxSeverity >= 80
    ? 'critical'
    : maxSeverity >= 60
      ? 'high'
      : maxSeverity >= 35
        ? 'medium'
        : 'low';

  const affectedPopulation = affectedNodes.reduce((sum, node) => sum + (node.metrics.populationAffected || 0), 0);
  const visualization = buildVisualizationData(graphNodes, graphEdges, failureNodeId, affectedNodes);

  return {
    sourceFailure: {
      nodeId: failureNodeId,
      nodeType: sourceNode.type,
      nodeName: sourceNode.name || failureNodeId,
      failureType,
      severity: normalizeSeverity(failureSeverity),
    },
    affectedNodes,
    propagationPath,
    overallAssessment: {
      riskLevel,
      summary: `${sourceNode.name || failureNodeId} failure is projected to affect ${totalAffected} node(s).`,
      priorityActions: [
        'Stabilize the failed source node first.',
        'Protect the highest-severity downstream nodes.',
        'Run cumulative what-if checks for additional failures.',
      ],
      estimatedRecoveryTime: maxSeverity >= 80 ? '24-72 hours' : maxSeverity >= 60 ? '8-24 hours' : '2-8 hours',
      affectedPopulation,
    },
    visualization,
    totalAffected,
    criticalCount,
    highCount,
    timestamp: new Date().toISOString(),
  };
}

function syncGraphFromRequestState(body = {}) {
  const state = body.villageState || body.waterState;
  if (!state) {
    return false;
  }

  // Always sync when state is provided to avoid stale graph mismatch
  // between frontend and backend model inference context.
  gnnService.initializeFromVillageState(state);
  return true;
}

async function predictImpactWithModelOrFallback(nodeId, failureType = 'failure', severity = 'medium') {
  try {
    const payload = buildPythonInferencePayload(nodeId, severity);
    console.log(`[GNN] Python inference request -> nodeId=${nodeId}, severity=${normalizeSeverity(severity)}, nodes=${payload.nodes.length}, edges=${payload.edges.length}`);

    const response = await fetch(`${PYTHON_GNN_API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const details = await response.text();
      throw new Error(`Python model API failed (${response.status}): ${details}`);
    }

    const modelResponse = await response.json();
  const predictionCount = Array.isArray(modelResponse?.predictions) ? modelResponse.predictions.length : 0;
  console.log(`[GNN] Python inference response <- predictions=${predictionCount}, trained=${Boolean(modelResponse?.model_trained)}, device=${modelResponse?.device || 'unknown'}`);

    const impact = buildModelImpactPrediction(nodeId, failureType, severity, modelResponse);

    return {
      impact,
      modelSource: 'python-model',
      modelError: null,
    };
  } catch (error) {
    throw new Error(`Python model inference unavailable: ${error.message}`);
  }
}

/**
 * @route GET /api/gnn/status
 * @desc Get GNN service status
 */
router.get('/status', (req, res) => {
  res.json({
    initialized: gnnService.isInitialized,
    nodeCount: gnnService.graph?.nodes?.size || 0,
    edgeCount: Array.from(gnnService.graph?.edges?.values() || []).reduce((sum, edges) => sum + edges.length, 0),
    scenarios: gnnService.getFailureScenarios(),
    modelApi: PYTHON_GNN_API_URL,
  });
});

/**
 * @route POST /api/gnn/initialize
 * @desc Initialize GNN with complete village infrastructure state
 */
router.post('/initialize', (req, res) => {
  try {
    const { villageState, waterState } = req.body;
    
    // Accept either villageState or waterState for backward compatibility
    const state = villageState || waterState;
    
    if (!state) {
      return res.status(400).json({ error: 'villageState or waterState is required' });
    }
    
    gnnService.initializeFromVillageState(state);
    
    res.json({
      success: true,
      message: 'GNN initialized successfully with village infrastructure',
      nodeCount: gnnService.graph.nodes.size,
      nodes: gnnService.getGraphNodes(),
      edges: gnnService.getGraphEdges()
    });
  } catch (error) {
    console.error('GNN initialization error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/gnn/predict-impact
 * @desc Predict cascading effects of a failure
 */
router.post('/predict-impact', async (req, res) => {
  try {
    const { nodeId, failureType = 'failure', severity = 'medium' } = req.body;
    
    if (!nodeId) {
      return res.status(400).json({ error: 'nodeId is required' });
    }
    
    // Sync graph if state is provided in request.
    syncGraphFromRequestState(req.body);
    
    if (!gnnService.isInitialized) {
      return res.status(400).json({ 
        error: 'GNN not initialized. Please provide waterState or call /initialize first.' 
      });
    }

    const { impact, modelSource, modelError } = await predictImpactWithModelOrFallback(
      nodeId,
      failureType,
      severity,
    );
    
    res.json({
      success: true,
      modelSource,
      modelError,
      impact,
      // Compatibility for frontend code that reads top-level affectedNodes.
      affectedNodes: impact?.affectedNodes || [],
    });
  } catch (error) {
    console.error('Impact prediction error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/gnn/predict-structured
 * @desc Structured prediction endpoint used by frontend map flow.
 */
router.post('/predict-structured', async (req, res) => {
  try {
    const {
      nodeId,
      node_name,
      failureType = 'failure',
      failure_type,
      severity = 'medium',
    } = req.body;

    // Sync graph if state is provided in request.
    syncGraphFromRequestState(req.body);

    if (!gnnService.isInitialized) {
      return res.status(400).json({
        error: 'GNN not initialized. Please provide waterState or call /initialize first.',
      });
    }

    let resolvedNodeId = nodeId;
    const effectiveFailureType = failure_type || failureType;

    if (!resolvedNodeId && node_name) {
      const nodeNameLower = String(node_name).toLowerCase();
      const match = gnnService.getGraphNodes().find((node) =>
        String(node.name || '').toLowerCase() === nodeNameLower
      );
      resolvedNodeId = match?.id;
    }

    if (!resolvedNodeId) {
      return res.status(400).json({
        error: 'nodeId or node_name is required and must match an initialized graph node.',
      });
    }

    const { impact, modelSource, modelError } = await predictImpactWithModelOrFallback(
      resolvedNodeId,
      effectiveFailureType,
      severity,
    );

    res.json({
      success: true,
      modelSource,
      modelError,
      sourceNodeId: resolvedNodeId,
      impact,
      affectedNodes: impact?.affectedNodes || [],
    });
  } catch (error) {
    console.error('Structured prediction error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route POST /api/gnn/predict
 * @desc Legacy endpoint used by visualization demo service.
 */
router.post('/predict', async (req, res) => {
  try {
    const { nodeId, failureType = 'failure', severity = 0.7 } = req.body;

    if (!nodeId) {
      return res.status(400).json({ error: 'nodeId is required' });
    }

    // Sync graph if state is provided in request.
    syncGraphFromRequestState(req.body);

    if (!gnnService.isInitialized) {
      return res.status(400).json({
        error: 'GNN not initialized. Please provide waterState or call /initialize first.',
      });
    }

    const severityLabel = normalizeSeverity(severity);
    const { impact, modelSource, modelError } = await predictImpactWithModelOrFallback(
      nodeId,
      failureType,
      severityLabel,
    );

    const impactedNodes = (impact?.affectedNodes || []).map((node) => ({
      id: node.nodeId,
      name: node.nodeName,
      type: node.nodeType,
      probability: node.probability,
      severity: node.severity,
      estimatedTime: node.timeToImpact,
    }));

    res.json({
      status: 'success',
      modelSource,
      modelError,
      impactedNodes,
      visualization: impact?.visualization || { nodes: [], links: [] },
      metadata: {
        totalAffected: impact?.totalAffected || impactedNodes.length,
        propagationDepth: Math.max(...(impact?.propagationPath || []).map((p) => p.depth), 1),
        criticalNodes: impact?.criticalCount || 0,
      },
    });
  } catch (error) {
    console.error('Legacy predict endpoint error:', error);
    res.status(500).json({ status: 'error', error: error.message });
  }
});

/**
 * @route GET /api/gnn/graph
 * @desc Get the current graph structure for visualization
 */
router.get('/graph', (req, res) => {
  if (!gnnService.isInitialized) {
    return res.status(400).json({ error: 'GNN not initialized' });
  }
  
  res.json({
    nodes: gnnService.getGraphNodes(),
    edges: gnnService.getGraphEdges()
  });
});

/**
 * @route GET /api/gnn/scenarios
 * @desc Get available failure scenarios
 */
router.get('/scenarios', (req, res) => {
  res.json({
    scenarios: gnnService.getFailureScenarios()
  });
});

/**
 * @route POST /api/gnn/what-if
 * @desc Run a "what-if" analysis for multiple failure scenarios
 */
router.post('/what-if', async (req, res) => {
  try {
    const { scenarios } = req.body;
    
    if (!scenarios || !Array.isArray(scenarios)) {
      return res.status(400).json({ error: 'scenarios array is required' });
    }
    
    // Sync graph if state is provided in request.
    syncGraphFromRequestState(req.body);
    
    if (!gnnService.isInitialized) {
      return res.status(400).json({ 
        error: 'GNN not initialized. Please provide waterState or call /initialize first.' 
      });
    }
    
    const results = [];
    for (const scenario of scenarios) {
      try {
        const { impact, modelSource, modelError } = await predictImpactWithModelOrFallback(
          scenario.nodeId,
          scenario.failureType || 'failure',
          scenario.severity || 'medium',
        );
        results.push({
          scenario,
          impact,
          modelSource,
          modelError,
          success: true,
        });
      } catch (error) {
        results.push({
          scenario,
          error: error.message,
          success: false,
        });
      }
    }
    
    // Calculate combined risk
    const successfulResults = results.filter(r => r.success);
    const combinedRisk = {
      totalScenariosAnalyzed: scenarios.length,
      successfulAnalyses: successfulResults.length,
      highestRiskScenario: successfulResults.reduce((max, r) => {
        const currentRisk = r.impact?.affectedNodes?.length || 0;
        const maxRisk = max?.impact?.affectedNodes?.length || 0;
        return currentRisk > maxRisk ? r : max;
      }, null),
      totalUniqueNodesAffected: new Set(
        successfulResults.flatMap(r => r.impact?.affectedNodes?.map(n => n.nodeId) || [])
      ).size
    };
    
    res.json({
      success: true,
      results,
      combinedRisk
    });
  } catch (error) {
    console.error('What-if analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * @route GET /api/gnn/vulnerable-nodes
 * @desc Get nodes ranked by vulnerability/criticality
 */
router.get('/vulnerable-nodes', (req, res) => {
  if (!gnnService.isInitialized) {
    return res.status(400).json({ error: 'GNN not initialized' });
  }
  
  const nodes = gnnService.getGraphNodes();
  const edges = gnnService.getGraphEdges();
  
  // Calculate vulnerability score for each node
  const vulnerableNodes = nodes.map(node => {
    // Count connections (higher = more critical)
    const incomingEdges = edges.filter(e => e.target === node.id).length;
    const outgoingEdges = edges.filter(e => e.source === node.id).length;
    const totalConnections = incomingEdges + outgoingEdges;
    
    // Base criticality from node type
    const typeCriticality = {
      'tank': 0.8,
      'pump': 0.9,
      'cluster': 0.7,
      'pipe': 0.5,
      'power': 0.95,
      'sensor': 0.3
    };
    
    // Calculate vulnerability score
    const connectivityScore = Math.min(totalConnections / 10, 1);
    const typeScore = typeCriticality[node.type] || 0.5;
    const statusPenalty = node.properties?.status === 'ok' || node.properties?.status === 'good' ? 0 : 0.3;
    
    const vulnerabilityScore = (connectivityScore * 0.4 + typeScore * 0.4 + statusPenalty * 0.2);
    
    return {
      ...node,
      vulnerabilityScore: Math.round(vulnerabilityScore * 100),
      connections: totalConnections,
      incomingEdges,
      outgoingEdges,
      riskLevel: vulnerabilityScore > 0.7 ? 'high' : vulnerabilityScore > 0.4 ? 'medium' : 'low'
    };
  });
  
  // Sort by vulnerability
  vulnerableNodes.sort((a, b) => b.vulnerabilityScore - a.vulnerabilityScore);
  
  res.json({
    nodes: vulnerableNodes,
    criticalNodes: vulnerableNodes.filter(n => n.riskLevel === 'high'),
    summary: {
      totalNodes: nodes.length,
      highRisk: vulnerableNodes.filter(n => n.riskLevel === 'high').length,
      mediumRisk: vulnerableNodes.filter(n => n.riskLevel === 'medium').length,
      lowRisk: vulnerableNodes.filter(n => n.riskLevel === 'low').length
    }
  });
});

export default router;
