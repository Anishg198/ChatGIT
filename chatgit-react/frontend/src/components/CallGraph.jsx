import React, { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import axios from 'axios';
import { API_BASE_URL } from '../config';

const CallGraph = () => {
    const [graphElements, setGraphElements] = useState({ nodes: [], links: [] });
    const [focusedNode, setFocusedNode] = useState('Show All');
    const [nodeList, setNodeList] = useState([]);
    const wrapperRef = useRef(null);
    const [size, setSize] = useState({ width: 800, height: 600 });

    useEffect(() => {
        // Fetch list of functions for dropdown
        axios.get('${API_BASE_URL}/api/call_graph')
            .then(res => {
                if (res.data.functions) setNodeList(['Show All', ...res.data.functions]);
            })
            .catch(err => console.error(err));
    }, []);

    useEffect(() => {
        const loadGraph = async () => {
            try {
                const res = await axios.post('${API_BASE_URL}/api/call_graph/visualize', { target: focusedNode });
                if (res.data.nodes) {
                    const connections = res.data.edges.map(e => ({
                        source: e.source,
                        target: e.target
                    }));
                    setGraphElements({ nodes: res.data.nodes, links: connections });
                }
            } catch (e) {
                console.error(e);
            }
        };
        loadGraph();
    }, [focusedNode]);

    useEffect(() => {
        if (wrapperRef.current) {
            setSize({
                width: wrapperRef.current.offsetWidth,
                height: 600
            });
        }

        const handleResize = () => {
            if (wrapperRef.current) {
                setSize({
                    width: wrapperRef.current.offsetWidth,
                    height: 600
                });
            }
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    return (
        <div ref={wrapperRef} className="call-graph-container" style={{ border: '1px solid #333', marginBottom: '20px', background: '#050505', borderRadius: '8px', overflow: 'hidden' }}>
            <div className="graph-controls" style={{ padding: '15px', display: 'flex', gap: '15px', borderBottom: '1px solid #333', alignItems: 'center' }}>
                <span style={{ fontWeight: 'bold' }}>CALL GRAPH VISUALIZATION</span>
                <select
                    value={focusedNode}
                    onChange={e => setFocusedNode(e.target.value)}
                    style={{
                        background: '#000',
                        color: '#fff',
                        border: '1px solid #333',
                        padding: '8px',
                        borderRadius: '4px',
                        minWidth: '200px'
                    }}
                >
                    {nodeList.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
                <div style={{ color: '#666', fontSize: '12px' }}>
                    {graphElements.nodes.length} nodes • {graphElements.links.length} edges
                </div>
            </div>

            <ForceGraph2D
                graphData={graphElements}
                nodeLabel="label"
                backgroundColor="#050505"
                width={size.width}
                height={size.height}
                linkDirectionalArrowLength={3.5}
                linkDirectionalArrowRelPos={1}
                nodeCanvasObject={(node, ctx, globalScale) => {
                    const label = node.label;
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Sans-Serif`;
                    const textWidth = ctx.measureText(label).width;
                    const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

                    // Draw Node
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, 4, 0, 2 * Math.PI, false);
                    ctx.fillStyle = node.id === focusedNode ? '#ff0000' : '#002AE0'; // Highlight target if matches
                    ctx.fill();

                    // Draw Label
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                    ctx.fillText(label, node.x, node.y + 6);

                    node.__bckgDimensions = bckgDimensions; // to re-use in nodePointerAreaPaint
                }}
                nodePointerAreaPaint={(node, color, ctx) => {
                    // This is needed for hover/click interactions to work correctly with custom canvas object
                    ctx.fillStyle = color;
                    const bckgDimensions = node.__bckgDimensions;
                    bckgDimensions && ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);
                }}
                linkColor={() => "#555"}
            />
        </div>
    );
};

export default CallGraph;
