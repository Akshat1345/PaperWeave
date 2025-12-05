# modules/graph_viz.py - Knowledge Graph Visualization
import os
import json
import networkx as nx
from typing import Dict, List, Optional
from config import config
from modules.utils import logger

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, visualization will be limited")

class GraphVisualizer:
    """Generate visualizations of the knowledge graph."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.export_dir = config.GRAPH_EXPORT_DIR
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_to_html(self, filename: str = "knowledge_graph.html", 
                      job_id: Optional[int] = None) -> str:
        """
        Export graph as interactive HTML using vis.js network.
        
        Args:
            filename: Output filename
            job_id: Filter to specific job
        
        Returns:
            Path to HTML file
        """
        try:
            # Filter graph by job if specified
            if job_id:
                subgraph = self._filter_by_job(job_id)
            else:
                subgraph = self.graph
            
            # Build nodes and edges data
            nodes = []
            edges = []
            
            # Process nodes
            for node_id in subgraph.nodes():
                node_data = subgraph.nodes[node_id]
                node_type = node_data.get('type', 'unknown')
                
                # Node configuration by type
                if node_type == 'paper':
                    color = '#4A90E2'
                    shape = 'box'
                    label = node_data.get('title', 'Unknown')[:40]
                    size = 30
                elif node_type == 'author':
                    color = '#F5A623'
                    shape = 'dot'
                    label = node_data.get('name', 'Unknown')
                    size = 20
                elif node_type == 'concept':
                    color = '#7ED321'
                    shape = 'diamond'
                    label = node_data.get('name', 'Unknown')
                    size = 15
                else:
                    color = '#CCCCCC'
                    shape = 'dot'
                    label = str(node_id)
                    size = 10
                
                nodes.append({
                    'id': node_id,
                    'label': label,
                    'color': color,
                    'shape': shape,
                    'size': size,
                    'title': self._node_tooltip(node_id, node_data),
                    'type': node_type
                })
            
            # Process edges
            for source, target, key, edge_data in subgraph.edges(keys=True, data=True):
                relationship = edge_data.get('relationship', 'related')
                
                # Edge style by relationship
                if relationship == 'cites':
                    color = '#4A90E2'
                    width = 2
                    arrows = 'to'
                elif relationship == 'authored':
                    color = '#F5A623'
                    width = 1
                    arrows = 'to'
                elif relationship == 'discusses':
                    color = '#7ED321'
                    width = 1
                    arrows = 'to'
                else:
                    color = '#CCCCCC'
                    width = 1
                    arrows = 'to'
                
                edges.append({
                    'from': source,
                    'to': target,
                    'color': color,
                    'width': width,
                    'arrows': arrows,
                    'title': relationship
                })
            
            # Generate HTML
            html_content = self._generate_html_template(nodes, edges, job_id)
            
            # Save file
            filepath = os.path.join(self.export_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Exported graph to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"HTML export error: {e}", exc_info=True)
            return None
    
    def _filter_by_job(self, job_id: int) -> nx.MultiDiGraph:
        """Filter graph to papers from specific job."""
        from modules.database import db
        
        papers = db.get_papers_by_job(job_id)
        paper_ids = {f"paper_{p['id']}" for p in papers}
        
        # Get subgraph with papers and their connections
        nodes_to_keep = set()
        
        for node in paper_ids:
            if self.graph.has_node(node):
                nodes_to_keep.add(node)
                # Add connected nodes (authors, concepts)
                nodes_to_keep.update(self.graph.neighbors(node))
                nodes_to_keep.update(self.graph.predecessors(node))
        
        return self.graph.subgraph(nodes_to_keep).copy()
    
    def _node_tooltip(self, node_id: str, node_data: Dict) -> str:
        """Generate tooltip HTML for node."""
        node_type = node_data.get('type', 'unknown')
        
        if node_type == 'paper':
            return f"""
<b>{node_data.get('title', 'Unknown')}</b><br>
ArXiv: {node_data.get('arxiv_id', 'N/A')}<br>
Citations: {node_data.get('citation_count', 0)}<br>
Year: {node_data.get('year', 'N/A')}
"""
        elif node_type == 'author':
            return f"<b>Author:</b> {node_data.get('name', 'Unknown')}"
        elif node_type == 'concept':
            return f"<b>Concept:</b> {node_data.get('name', 'Unknown')}"
        else:
            return str(node_id)
    
    def _generate_html_template(self, nodes: List[Dict], edges: List[Dict], 
                               job_id: Optional[int]) -> str:
        """Generate complete HTML with vis.js."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #f5f7fa;
        }}
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        #controls {{
            padding: 15px;
            background: white;
            border-bottom: 2px solid #e0e0e0;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }}
        button {{
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{ background: #5568d3; }}
        #network {{
            width: 100%;
            height: calc(100vh - 150px);
            border: 1px solid #ddd;
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        #stats {{
            padding: 10px 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
            color: #555;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🕸️ Research Knowledge Graph</h1>
        <p>{f'Job {job_id}' if job_id else 'All Papers'}</p>
    </div>
    
    <div id="stats">
        <strong>Nodes:</strong> <span id="nodeCount">{len(nodes)}</span> |
        <strong>Edges:</strong> <span id="edgeCount">{len(edges)}</span> |
        <strong>Papers:</strong> <span id="paperCount">{len([n for n in nodes if n['type'] == 'paper'])}</span>
    </div>
    
    <div id="controls">
        <button onclick="network.fit()">🔍 Fit View</button>
        <button onclick="showPapersOnly()">📄 Papers Only</button>
        <button onclick="showAll()">🌐 Show All</button>
        <button onclick="exportJSON()">💾 Export JSON</button>
    </div>
    
    <div id="network"></div>
    
    <div class="legend">
        <h4>Legend</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: #4A90E2;"></div>
            <span>Paper</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #F5A623;"></div>
            <span>Author</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #7ED321;"></div>
            <span>Concept</span>
        </div>
    </div>
    
    <script>
        // Network data
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        
        // Create network
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                font: {{ size: 14, color: '#333' }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                shadow: true
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -2000,
                    springConstant: 0.001,
                    springLength: 200
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        
        // Event handlers
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Clicked node:', node);
            }}
        }});
        
        function showPapersOnly() {{
            nodes.forEach(node => {{
                if (node.type === 'paper') {{
                    nodes.update({{id: node.id, hidden: false}});
                }} else {{
                    nodes.update({{id: node.id, hidden: true}});
                }}
            }});
            network.fit();
        }}
        
        function showAll() {{
            nodes.forEach(node => {{
                nodes.update({{id: node.id, hidden: false}});
            }});
            network.fit();
        }}
        
        function exportJSON() {{
            const graphData = {{
                nodes: nodes.get(),
                edges: edges.get()
            }};
            const blob = new Blob([JSON.stringify(graphData, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'knowledge_graph.json';
            a.click();
        }}
    </script>
</body>
</html>"""
    
    def export_to_json(self, filename: str = "knowledge_graph.json") -> str:
        """Export graph as JSON."""
        try:
            data = {
                'nodes': [],
                'edges': []
            }
            
            # Nodes
            for node_id in self.graph.nodes():
                node_data = dict(self.graph.nodes[node_id])
                node_data['id'] = node_id
                data['nodes'].append(node_data)
            
            # Edges
            for source, target, key, edge_data in self.graph.edges(keys=True, data=True):
                edge = dict(edge_data)
                edge['source'] = source
                edge['target'] = target
                data['edges'].append(edge)
            
            filepath = os.path.join(self.export_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported JSON to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"JSON export error: {e}")
            return None

def visualize_graph(graph: nx.MultiDiGraph, job_id: Optional[int] = None) -> str:
    """
    Convenience function to visualize graph.
    
    Returns path to HTML file.
    """
    viz = GraphVisualizer(graph)
    return viz.export_to_html(job_id=job_id)