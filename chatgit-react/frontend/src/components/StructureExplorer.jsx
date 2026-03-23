import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

const TreeNode = ({ name, data, isFile, depth }) => {
    const [expanded, setExpanded] = useState(false);

    const toggle = () => setExpanded(!expanded);

    const paddingLeft = `${depth * 20}px`;

    // Icon logic
    let icon = "📁";
    if (isFile) {
        if (name.endsWith('.py')) icon = "🐍";
        else if (name.endsWith('.js')) icon = "📜";
        else if (name.endsWith('.ts')) icon = "📘";
        else icon = "📄";
    }

    // Render File Details (Functions/Classes)
    if (isFile && expanded) {
        const functions = data.functions || [];
        const classes = data.classes || [];

        return (
            <div>
                <div
                    onClick={toggle}
                    style={{
                        paddingLeft,
                        cursor: 'pointer',
                        userSelect: 'none',
                        paddingTop: '4px',
                        paddingBottom: '4px',
                        backgroundColor: expanded ? '#E6E6FF' : 'transparent',
                        color: expanded ? '#0000FF' : 'inherit'
                    }}
                >
                    {icon} {name}
                </div>
                <div style={{ marginLeft: `${depth * 20 + 20}px`, borderLeft: '1px solid #CCC', paddingLeft: '10px' }}>
                    {classes.map((c, i) => (
                        <div key={`c-${i}`} style={{ fontSize: '0.85em', color: '#6A00FF' }}>
                            Box: {c.name}
                        </div>
                    ))}
                    {functions.map((f, i) => (
                        <div key={`f-${i}`} style={{ fontSize: '0.85em', color: '#008000' }}>
                            ƒ: {f.name}
                        </div>
                    ))}
                    {classes.length === 0 && functions.length === 0 && (
                        <div style={{ fontSize: '0.8em', color: '#888', fontStyle: 'italic' }}>
                            (No parsed symbols)
                        </div>
                    )}
                </div>
            </div>
        );
    }

    if (!isFile && expanded) {
        return (
            <div>
                <div
                    onClick={toggle}
                    style={{
                        paddingLeft,
                        cursor: 'pointer',
                        userSelect: 'none',
                        fontWeight: 'bold',
                        paddingTop: '4px',
                        paddingBottom: '4px'
                    }}
                >
                    {expanded ? '📂' : '📁'} {name}
                </div>
                {Object.keys(data).sort().map((childName) => {
                    const isChildFile = data[childName].__isFile;
                    return (
                        <TreeNode
                            key={childName}
                            name={childName}
                            data={data[childName]}
                            isFile={isChildFile}
                            depth={depth + 1}
                        />
                    );
                })}
            </div>
        );
    }

    // Collapsed state
    return (
        <div
            onClick={toggle}
            style={{
                paddingLeft,
                cursor: 'pointer',
                userSelect: 'none',
                paddingTop: '4px',
                paddingBottom: '4px',
                fontWeight: isFile ? 'normal' : 'bold'
            }}
        >
            {isFile ? icon : '📁'} {name}
        </div>
    );
};

const StructureExplorer = () => {
    const [fileTree, setFileTree] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStructure = async () => {
            try {
                const res = await axios.get(`${API_BASE_URL}/api/structure`);
                const rawFiles = res.data; // Dict of "path/to/file": { metadata }

                // Build Tree
                const treeRoot = {};

                Object.keys(rawFiles).forEach(path => {
                    const parts = path.split('/');
                    let currentLevel = treeRoot;

                    parts.forEach((part, index) => {
                        if (index === parts.length - 1) {
                            // File Node
                            currentLevel[part] = { ...rawFiles[path], __isFile: true };
                        } else {
                            // Directory Node
                            if (!currentLevel[part]) {
                                currentLevel[part] = {};
                            }
                            currentLevel = currentLevel[part];
                        }
                    });
                });

                setFileTree(treeRoot);
            } catch (err) {
                console.error("Failed to fetch structure", err);
            } finally {
                setLoading(false);
            }
        };

        fetchStructure();
    }, []);

    if (loading) return <div>Loading file system...</div>;
    if (!fileTree) return <div>No structure available.</div>;

    return (
        <div className="structure-explorer-container" style={{
            border: '2px solid #0000FF',
            padding: '20px',
            margin: '20px 0',
            backgroundColor: '#FFFFFF',
            maxHeight: '600px',
            overflowY: 'auto'
        }}>
            <h2 style={{ marginBottom: '15px' }}>REPOSITORY EXPLORER (AST View)</h2>
            <div className="tree-root">
                {Object.keys(fileTree).sort().map(name => (
                    <TreeNode
                        key={name}
                        name={name}
                        data={fileTree[name]}
                        isFile={fileTree[name].__isFile}
                        depth={0}
                    />
                ))}
            </div>
        </div>
    );
};

export default StructureExplorer;
