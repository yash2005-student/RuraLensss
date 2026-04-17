/**
 * GNN Impact Service - Frontend API Client
 * Connects to the backend GNN service for infrastructure impact prediction
 */

import type { GraphVisualizationData } from '../types/graph-visualization';
import { API_URL } from '../config/api';

export interface ImpactAnalysisRequest {
  nodeId: string;
  severity?: number;
  timestamp?: Date;
}

export interface ImpactAnalysisResponse {
  status: 'success' | 'error';
  impactedNodes: Array<{
    id: string;
    name: string;
    type: string;
    probability: number;
    severity: string;
    estimatedTime: number;
  }>;
  visualization: GraphVisualizationData;
  metadata?: {
    totalAffected: number;
    propagationDepth: number;
    criticalNodes: number;
  };
}

class GNNImpactService {
  private baseUrl: string;

  constructor(baseUrl: string = API_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Predict infrastructure failure impact
   */
  async predictImpact(request: ImpactAnalysisRequest): Promise<ImpactAnalysisResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/gnn/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error predicting impact:', error);
      throw error;
    }
  }

  /**
   * Get the current infrastructure graph state
   */
  async getInfrastructureGraph(): Promise<GraphVisualizationData> {
    try {
      const response = await fetch(`${this.baseUrl}/api/gnn/graph`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.visualization || data;
    } catch (error) {
      console.error('Error fetching infrastructure graph:', error);
      throw error;
    }
  }

  /**
   * Initialize the GNN service with village state
   */
  async initialize(villageState: any): Promise<{ success: boolean }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/gnn/initialize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ villageState }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error initializing GNN service:', error);
      throw error;
    }
  }
}

// Singleton instance
export const gnnService = new GNNImpactService();

export default GNNImpactService;
